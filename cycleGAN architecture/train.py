from torchvision.utils import save_image
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/Individual signature/train"
VAL_DIR = "data/Individual signature/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 1
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_F = "genf.pth.tar"
CHECKPOINT_GEN_O = "geno.pth.tar"
CHECKPOINT_CRITIC_F = "criticf.pth.tar"
CHECKPOINT_CRITIC_O = "critico.pth.tar"

transforms = A.Compose(
    [
        # A.Resize(width=224, height=224),
        A.HorizontalFlip(p=0.5),
        # Assuming single-channel grayscale
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)


def train_fn(
    disc_forg, disc_org, gen_org, gen_forg, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    F_real = 0
    F_fake = 0
    loop = tqdm(loader, leave=True)

    for idx, (org, forg, org_path, forg_path) in enumerate(loop):
        org = org.to(DEVICE)
        forg = forg.to(DEVICE)

        # print(f"Original Image {org.shape}")
        # print(f"Forgery Image {forg.shape}")
        # print(f"Original Path {org_path}")
        # print(f"Forged Path {forg_path}")
        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
            fake_forg = gen_forg(org)
            D_F_real = disc_forg(forg)
            D_F_fake = disc_forg(fake_forg.detach())
            F_real += D_F_real.mean().item()
            F_fake += D_F_fake.mean().item()
            D_F_real_loss = mse(D_F_real, torch.ones_like(D_F_real))
            D_F_fake_loss = mse(D_F_fake, torch.zeros_like(D_F_fake))
            D_F_loss = D_F_real_loss + D_F_fake_loss

            fake_org = gen_org(forg)
            D_O_real = disc_org(org)
            D_O_fake = disc_org(fake_org.detach())
            D_O_real_loss = mse(D_O_real, torch.ones_like(D_O_real))
            D_O_fake_loss = mse(D_O_fake, torch.zeros_like(D_O_fake))
            D_O_loss = D_O_real_loss + D_O_fake_loss

            # put it togethor
            D_loss = (D_F_loss + D_O_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_F_fake = disc_forg(fake_forg)
            D_O_fake = disc_org(fake_org)
            loss_G_H = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_Z = mse(D_O_fake, torch.ones_like(D_O_fake))

            # cycle loss
            cycle_org = gen_org(fake_forg)
            cycle_forg = gen_forg(fake_org)
            cycle_org_loss = l1(org, cycle_org)
            cycle_forg_loss = l1(forg, cycle_forg)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_org = gen_org(org)
            identity_forg = gen_forg(forg)
            identity_org_loss = l1(org, identity_org)
            identity_forg_loss = l1(forg, identity_forg)

            # add all togethor
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_org_loss * LAMBDA_CYCLE
                + cycle_forg_loss * LAMBDA_CYCLE
                + identity_forg_loss * LAMBDA_IDENTITY
                + identity_org_loss * LAMBDA_IDENTITY
            )
        print("Loss: " + str(G_loss))
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            # save_image(fake_forg * 0.5 + 0.5, f"saved_images/org_{idx}.png")
            # save_image(fake_org * 0.5 + 0.5, f"saved_images/forg{idx}.png")
            save_image(fake_forg * 0.5 + 0.5,
                       f"saved_images/{os.path.basename(str(org_path[0]))}.png")
            save_image(fake_org * 0.5 + 0.5,
                       f"saved_images/{os.path.basename(str(forg_path[0]))}.png")

        loop.set_postfix(H_real=F_real / (idx + 1), H_fake=F_fake / (idx + 1))


def main():
    disc_forg = Discriminator(in_channels=1).to(DEVICE)
    disc_org = Discriminator(in_channels=1).to(DEVICE)
    gen_org = Generator(img_channels=1, num_residuals=9).to(DEVICE)
    gen_forg = Generator(img_channels=1, num_residuals=9).to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_forg.parameters()) + list(disc_org.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_org.parameters()) + list(gen_forg.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if LOAD_MODEL:
        load_checkpoint(
            CHECKPOINT_GEN_F,
            gen_forg,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_GEN_O,
            gen_org,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC_F,
            disc_forg,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            CHECKPOINT_CRITIC_O,
            disc_org,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = SignatureDataset(
        root_forg="data/Individual signature/trainA",
        root_org="data/Individual signature/trainB",
        transform=transforms,
    )
    val_dataset = SignatureDataset(
        root_forg="data/Individual signature/testA",
        root_org="data/Individual signature/testB",
        transform=transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(
            disc_forg,
            disc_org,
            gen_org,
            gen_forg,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if SAVE_MODEL:
            save_checkpoint(gen_forg, opt_gen, filename=CHECKPOINT_GEN_F)
            save_checkpoint(gen_org, opt_gen, filename=CHECKPOINT_GEN_O)
            save_checkpoint(disc_forg, opt_disc, filename=CHECKPOINT_GEN_F)
            save_checkpoint(disc_org, opt_disc, filename=CHECKPOINT_GEN_O)


if __name__ == "__main__":
    main()

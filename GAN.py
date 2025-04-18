import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
image_size = 28*28
batch_size = 100
sample_dir = 'samples'
experiments = [
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002},
    {"latent_size": 128, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002},
    {"latent_size": 256, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002},
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 25, "lr": 0.00015},
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 30, "lr": 0.0001},
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002}, #nn.Sigmoid()
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002}, #nn.Softsign()
    {"latent_size": 64, "hidden_size": 256, "num_epochs": 20, "lr": 0.0002}  #minimizing log(1-D(G(z)))
]

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],   # 1 for greyscale channels
                                     std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.FashionMNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    ])

inception_score = InceptionScore(normalize=True).to(device)
fid_score = FrechetInceptionDistance(normalize=True).to(device)

for exp_num, config in enumerate(experiments):
    latent_size = config["latent_size"]
    hidden_size = config["hidden_size"]
    num_epochs = config["num_epochs"]
    lr = config["lr"]
# Discriminator
    D = nn.Sequential(
                nn.Linear(image_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid())

    # Generator
    G = nn.Sequential(
        nn.Linear(latent_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh()) 
    
    if exp_num==5:
            print("exp_num:{}".format(exp_num))
            G = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Sigmoid()) #nn.Sigmoid(),不要denorm
            
    if exp_num==6:
            print("exp_num:{}".format(exp_num))
            G = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Softsign()) 

    D = D.to(device)
    G = G.to(device)

    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr) 
    g_optimizer = torch.optim.Adam(G.parameters(), lr)

    def denorm(x):
        # TANH 输出范围(-1, 1)
        out = (x + 1) / 2
        return out.clamp(0, 1) #torch.Tensor.clamp

    def reset_grad():
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

    # Start training
    total_step = len(data_loader)
    ii=torch.tensor([0.])
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.reshape(batch_size, -1).to(device)

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ================================================================== #
            #                      Train the discriminator                       #
            # ================================================================== #

            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop and optimize
            d_loss = d_loss_real + d_loss_fake
            reset_grad()
            # calculates gradient
            d_loss.backward()
            # Update parameters
            d_optimizer.step()

            # ================================================================== #
            #                        Train the generator                         #
            # ================================================================== #

            # Compute loss with fake images
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels) 
            
            if exp_num==7:
             print("exp_num:{}".format(exp_num))
             g_loss = -criterion(outputs, fake_labels)

            # Backprop and optimize
            reset_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                    .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                            real_score.mean().item(), fake_score.mean().item()))


            # Save sample images and model after each epoch
        if (epoch + 1) % 5 == 0:
                fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
                if exp_num!=5:
                 save_image(denorm(fake_images), os.path.join(sample_dir, f'fake_images_exp{exp_num+1}_epoch{epoch+1}.png'))
                else:
                 save_image(fake_images, os.path.join(sample_dir, f'fake_images_exp{exp_num+1}_epoch{epoch+1}.png'))

            # Save real images
        if (epoch+1) == 1:
                images = images.reshape(images.size(dim=0), 1, 28, 28)
                save_image(denorm(images), os.path.join(sample_dir, f'real_images_exp{exp_num+1}.png'))
                ii=denorm(images)
    

        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z).reshape(batch_size, 1, 28, 28)
        # Transform generated images for IS and FID calculation
        real_images_rgb = eval_transform(ii.reshape(ii.size(dim=0), 1, 28, 28)).expand(-1, 3, 28, 28)
        if exp_num!=5:
            fake_images_rgb = eval_transform(denorm(fake_images)).expand(-1, 3, 28, 28) #eval_transform(fake_images)返回tensor   
        else:
            fake_images_rgb = eval_transform(fake_images).expand(-1, 3, 28, 28) #eval_transform(fake_images)返回tensor 
        # Update Inception and FID metrics
        inception_score.update(fake_images_rgb)
        fid_score.update(real_images_rgb, real=True)
        fid_score.update(fake_images_rgb, real=False)
        # Calculate and save IS and FID at the end of each experiment
        is_score,_ = inception_score.compute()
        fid = fid_score.compute()
        print(f"Epoch [{epoch}/{num_epochs}], Inception Score: {is_score}, FID Score: {fid}")
        inception_score.reset()
        fid_score.reset()


    print(f"Experiment {exp_num+1} completed with latent_size={latent_size}, hidden_size={hidden_size}, "
          f"num_epochs={num_epochs}, learning_rate={lr}")
    print(f"Inception Score for Experiment {exp_num+1}: {is_score}")
    print(f"FID Score for Experiment {exp_num+1}: {fid}")

    with open("experiment_results.txt", "a") as file:
        file.write(f"Experiment {exp_num+1}:\n")
        file.write(f"latent_size={latent_size}, hidden_size={hidden_size}, num_epochs={num_epochs}, lr={lr}\n")
        file.write(f"Final d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}\n")
        file.write(f"Final Inception Score: {is_score}\n")
        file.write(f"Final FID Score: {fid}\n\n")

    inception_score.reset()
    fid_score.reset()



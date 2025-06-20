if __name__ == "__main__":
    from dataset import FingerprintDataset, get_fingerprint_transforms
    from torch.utils.data import DataLoader
    import os
    from net import FingerprintNet
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    # DATA
    image_target_size = (224, 224)

    train_transforms = get_fingerprint_transforms(
        image_size=image_target_size,
        apply_rotate=True,
        rotate_limit=15,
        apply_shift_scale_rotate=True,
        shift_limit=0.06,
        scale_limit=0.15,
        apply_horizontal_flip=True,
        apply_vertical_flip=False,
        apply_brightness_contrast=False,
        apply_gaussian_noise=False,
        normalize_image=True
    )

    val_test_transforms = get_fingerprint_transforms(
        image_size=image_target_size,
        apply_rotate=False,
        apply_shift_scale_rotate=False,
        apply_horizontal_flip=False,
        apply_vertical_flip=False,
        apply_brightness_contrast=False,
        apply_gaussian_noise=False,
        normalize_image=True
    )

    split_data_json = "data_split.json"

    train_dataset = FingerprintDataset(split_data_json, "train", transform=train_transforms, target_image_size=image_target_size)
    val_dataset = FingerprintDataset(split_data_json, "val", transform=val_test_transforms, target_image_size=image_target_size)
    test_dataset = FingerprintDataset(split_data_json, "test", transform=val_test_transforms, target_image_size=image_target_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUżywam urządzenia: {device}")
    if device.type == 'cuda':
        print(f"Nazwa GPU: {torch.cuda.get_device_name(0)}")

    num_workers_to_use = os.cpu_count() // 2 or 1
    if os.name == 'nt': # Sprawdź, czy system to Windows
        print("Wykryto Windows. Ustawiam num_workers=0, aby uniknąć problemów z multiprocessing.")
        num_workers_to_use = 0

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers_to_use, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)

    print(f"\nLiczba próbek w DataLoaderach:")
    print(f"  Treningowy: {len(train_loader.dataset)} (batchy: {len(train_loader)})")
    print(f"  Walidacyjny: {len(val_loader.dataset)} (batchy: {len(val_loader)})")
    print(f"  Testowy: {len(test_loader.dataset)} (batchy: {len(test_loader)})")

    # MODEL
    model = FingerprintNet("resnet50", True)
    model.to(device)

    loss_fn_coords = nn.MSELoss()
    loss_fn_delta_existence = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # TRAIN
    num_epochs = 100
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = -1
    model_save_path = "best_fingerprint_model.pth"

    train_losses_per_epoch: list[float] = []
    val_losses_per_epoch: list[float] = []

    patience = 10
    epochs_no_improve = 0

    total_training_start_time = time.time()

    print(f"\nRozpoczynanie treningu na {num_epochs} epokach...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time() # Czas rozpoczęcia epoki

        model.train()
        running_train_loss = 0.0

        for batch_idx, (images, core_coords_gt, delta_coords_gt, delta_existence_gt) in enumerate(train_loader):
            images = images.to(device)
            core_coords_gt = core_coords_gt.to(device)
            delta_coords_gt = delta_coords_gt.to(device)
            delta_existence_gt = delta_existence_gt.to(device)

            optimizer.zero_grad()

            core_preds, delta_coords_preds, delta_existence_logits = model(images)

            loss_core = loss_fn_coords(core_preds, core_coords_gt)
            loss_delta_existence = loss_fn_delta_existence(delta_existence_logits.squeeze(1), delta_existence_gt)

            delta_mask = delta_existence_gt.unsqueeze(1).expand_as(delta_coords_preds) # (batch_size, 1) -> (batch_size, 2)
            
            raw_delta_coords_loss = loss_fn_coords(delta_coords_preds, delta_coords_gt)
            
            if delta_mask.sum() > 0:
                loss_delta_coords = (raw_delta_coords_loss * delta_mask).sum() / delta_mask.sum()
            else:
                loss_delta_coords = torch.tensor(0.0, device=device) # Brak delt w batchu, strata 0

            total_loss = loss_core + loss_delta_coords + loss_delta_existence

            total_loss.backward()
            optimizer.step()

            running_train_loss += total_loss.item()

            if (batch_idx + 1) % 50 == 0: # Drukuj co 50 batchy
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses_per_epoch.append(avg_train_loss)

        train_epoch_end_time = time.time()
        train_epoch_duration = train_epoch_end_time - epoch_start_time
        print(f"--- Epoch {epoch+1} Treningowa Strata: {avg_train_loss:.4f} (Czas treningu: {train_epoch_duration:.2f}s) ---")

        # WALIDACJA
        val_start_time = time.time()
        model.eval()
        running_val_loss = 0.0

        correct_delta_predictions = 0
        total_delta_predictions = 0

        with torch.no_grad():
            for images, core_coords_gt, delta_coords_gt, delta_existence_gt in val_loader:
                images = images.to(device)
                core_coords_gt = core_coords_gt.to(device)
                delta_coords_gt = delta_coords_gt.to(device)
                delta_existence_gt = delta_existence_gt.to(device)

                core_preds, delta_coords_preds, delta_existence_logits = model(images)

                loss_core = loss_fn_coords(core_preds, core_coords_gt)
                loss_delta_existence = loss_fn_delta_existence(delta_existence_logits.squeeze(1), delta_existence_gt)

                delta_mask = delta_existence_gt.unsqueeze(1).expand_as(delta_coords_preds)
                raw_delta_coords_loss = loss_fn_coords(delta_coords_preds, delta_coords_gt)
                if delta_mask.sum() > 0:
                    loss_delta_coords = (raw_delta_coords_loss * delta_mask).sum() / delta_mask.sum()
                else:
                    loss_delta_coords = torch.tensor(0.0, device=device)

                total_loss = loss_core + loss_delta_coords + loss_delta_existence
                running_val_loss += total_loss.item()

                delta_predicted_prob = torch.sigmoid(delta_existence_logits).squeeze(1)
                delta_predicted_labels = (delta_predicted_prob > 0.5).float()
                
                correct_delta_predictions += (delta_predicted_labels == delta_existence_gt).sum().item()
                total_delta_predictions += delta_existence_gt.numel()

        avg_val_loss = running_val_loss / len(val_loader)
        val_delta_accuracy = correct_delta_predictions / total_delta_predictions if total_delta_predictions > 0 else 0.0
        val_losses_per_epoch.append(avg_val_loss)

        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        print(f"--- Epoch {epoch+1} Walidacyjna Strata: {avg_val_loss:.4f}, Dokładność Delty: {val_delta_accuracy:.4f} (Czas walidacji: {val_duration:.2f}s) ---")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss and avg_train_loss < best_train_loss:
            print(f"Walidacyjna strata poprawiła się z {best_val_loss:.4f} do {avg_val_loss:.4f}. Zapisuję model...")
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0

            best_epoch = epoch
        else:
            epochs_no_improve += 1
            print(f"Walidacyjna strata nie poprawiła się. Cierpliwość: {epochs_no_improve}/{patience}")
            if epochs_no_improve == patience:
                print(f"Brak poprawy przez {patience} epok. Zatrzymuję trening.")
                break

        epoch_end_time = time.time() # Całkowity czas trwania epoki
        epoch_total_duration = epoch_end_time - epoch_start_time
        print(f"--- Całkowity czas epoki {epoch+1}: {epoch_total_duration:.2f} sekund ---\n")

    total_training_end_time = time.time() # Czas zakończenia całego treningu
    total_training_duration = total_training_end_time - total_training_start_time
    print(f"\nTrening zakończony! Całkowity czas treningu: {total_training_duration:.2f} sekund")

    # SHOW LOSSES GRAPH
    epochs = np.arange(1, len(train_losses_per_epoch) + 1).astype(np.int32)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss")
    plt.plot(epochs, train_losses_per_epoch)
    plt.scatter(best_epoch + 1, best_train_loss, c='blue', s=15)

    plt.subplot(1, 2, 2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validate Loss")
    plt.plot(epochs, val_losses_per_epoch)
    plt.scatter(best_epoch + 1, best_val_loss, c='blue', s=15)

    plt.show()

    print(f"\nRozpoczynanie oceny na zbiorze testowym...")
    test_start_time = time.time()

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    running_test_loss = 0.0
    correct_test_delta_predictions = 0
    total_test_delta_predictions = 0

    with torch.no_grad():
        for images, core_coords_gt, delta_coords_gt, delta_existence_gt in test_loader:
            images = images.to(device)
            core_coords_gt = core_coords_gt.to(device)
            delta_coords_gt = delta_coords_gt.to(device)
            delta_existence_gt = delta_existence_gt.to(device)

            core_coords_preds, delta_coords_preds, delta_existence_logits = model(images)

            loss_core = loss_fn_coords(core_coords_preds, core_coords_gt)
            loss_delta_existence = loss_fn_delta_existence(delta_existence_logits.squeeze(1), delta_existence_gt)

            delta_mask = delta_existence_gt.unsqueeze(1).expand_as(delta_coords_preds)
            raw_delta_coords_loss = loss_fn_coords(delta_coords_preds, delta_coords_gt)
            if delta_mask.sum() > 0:
                loss_delta_coords = (raw_delta_coords_loss * delta_mask).sum() / delta_mask.sum()
            else:
                loss_delta_coords = torch.tensor(0.0, device=device)

            total_loss = loss_core + loss_delta_coords + loss_delta_existence
            running_test_loss += total_loss.item()
            
            delta_predicted_prob = torch.sigmoid(delta_existence_logits).squeeze(1)
            delta_predicted_labels = (delta_predicted_prob > 0.5).float()
            
            correct_test_delta_predictions += (delta_predicted_labels == delta_existence_gt).sum().item()
            total_test_delta_predictions += delta_existence_gt.numel()

            images_cpu = images.cpu()
            core_coords_gt_cpu = core_coords_gt.cpu()
            delta_coords_gt_cpu = delta_coords_gt.cpu()
            core_coords_preds_cpu = core_coords_preds.cpu()
            delta_coords_preds_cpu = delta_coords_preds.cpu()

            for idx, image in enumerate(images_cpu):
                transposed_image = np.transpose(image, (1, 2, 0))

                plt.figure(figsize=(5, 5))
                plt.imshow(transposed_image)

                width, height, _ = transposed_image.shape

                plt.scatter(core_coords_gt_cpu[idx][0] * width, core_coords_gt_cpu[idx][1] * height, c='blue', s=15)
                if delta_existence_gt[idx]:
                    plt.scatter(delta_coords_gt_cpu[idx][0] * width, delta_coords_gt_cpu[idx][1] * height, c='red', s=15)

                plt.scatter(core_coords_preds_cpu[idx][0] * width, core_coords_preds_cpu[idx][1] * height, c='green', s=15)
                if delta_predicted_labels[idx]:
                    plt.scatter(delta_coords_preds_cpu[idx][0] * width, delta_coords_preds_cpu[idx][1] * height, c='orange', s=15)

    avg_test_loss = running_test_loss / len(test_loader)
    test_delta_accuracy = correct_test_delta_predictions / total_test_delta_predictions if total_test_delta_predictions > 0 else 0.0

    test_end_time = time.time()
    test_duration = test_end_time - test_start_time
    print(f"--- Ostateczna Strata Testowa: {avg_test_loss:.4f}, Ostateczna Dokładność Delty: {test_delta_accuracy:.4f} (Czas testowania: {test_duration:.2f}s) ---")

    plt.show()
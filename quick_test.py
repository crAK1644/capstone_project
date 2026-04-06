"""
Quick SSFL test — 3 clients, 2 rounds, small epochs.
Validates the entire pipeline end-to-end without waiting 20+ minutes.
"""
import time
import numpy as np
from src.data import get_num_clients, get_num_open_samples, load_split
from src.client_app import SSFLClient
from src.train import evaluate_model
from src.utils import compute_metrics, get_device, set_seed

def main():
    set_seed(42)
    device = get_device()
    scenario = 1
    num_rounds = 2
    num_test_clients = 3  # Only use first 3 clients for speed
    lr = 0.0001
    batch_size = 100
    local_epochs = 1  # Reduced from 5 to 1 for speed
    num_open = get_num_open_samples()

    print(f"\n{'='*60}")
    print(f"  SSFL Quick Test — {num_test_clients} clients, {num_rounds} rounds")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # Create clients
    print("  Creating clients...", end="", flush=True)
    clients = []
    for cid in range(num_test_clients):
        c = SSFLClient(
            client_id=cid, scenario=scenario,
            lr=lr, local_epochs=local_epochs,
            discriminator_epochs=1, distillation_epochs=1,
            batch_size=batch_size, seed=42,
        )
        clients.append(c)
    print(" done.\n")

    # Test set
    test_loader = load_split("test", batch_size=batch_size, shuffle=False)

    global_labels = None
    start = time.time()

    for rnd in range(1, num_rounds + 1):
        print(f"  ── Round {rnd}/{num_rounds} ──")
        all_preds = []
        for i, client in enumerate(clients):
            if global_labels is not None:
                params = [global_labels.astype(np.float64)]
            else:
                params = [np.array([], dtype=np.float64)]
            config = {"server_round": rnd}
            t0 = time.time()
            result_params, n_samples, metrics = client.fit(params, config)
            hard_labels = result_params[0].astype(np.int16)
            all_preds.append(hard_labels)
            elapsed_c = time.time() - t0
            print(f"    Client {i}: cls_loss={metrics['classifier_loss']:.4f}, "
                  f"familiar={metrics['num_familiar']}/{num_open}, "
                  f"time={elapsed_c:.1f}s")

        # Voting
        pred_matrix = np.stack(all_preds, axis=0)
        global_labels = np.full(num_open, -1, dtype=np.int16)
        for j in range(num_open):
            votes = pred_matrix[:, j]
            valid = votes[votes >= 0]
            if len(valid) > 0:
                global_labels[j] = np.argmax(np.bincount(valid, minlength=11))

        num_labelled = int(np.sum(global_labels >= 0))

        # Eval
        loss, y_true, y_pred = evaluate_model(clients[0].classifier, test_loader, device)
        m = compute_metrics(y_true, y_pred)

        elapsed = time.time() - start
        print(f"    Voting: {num_labelled}/{num_open} labelled")
        print(f"    Test Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}")
        print(f"    Elapsed: {elapsed:.1f}s\n")

    print(f"{'='*60}")
    print(f"  ✅ SSFL Quick Test PASSED")
    print(f"  Final Accuracy: {m['accuracy']:.4f}")
    print(f"  Final F1:       {m['f1']:.4f}")
    print(f"  Total Time:     {time.time()-start:.1f}s")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

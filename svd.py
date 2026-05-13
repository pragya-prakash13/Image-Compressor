import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
size = 256
img = np.zeros((size, size), dtype=np.float64)
for i in range(size):
    for j in range(size):
        img[i, j] = 0.3*(i/size) + 0.2*(j/size)
        if (i-80)**2 + (j-80)**2 < 40**2:
            img[i, j] += 0.5
        if (i-170)**2 + (j-160)**2 < 55**2:
            img[i, j] += 0.4
        if 100 < i < 180 and 10 < j < 60:
            img[i, j] += 0.6
img = np.clip(img + np.random.randn(size, size)*0.05, 0, 1)

m, n = img.shape
print(f"Image shape: {m} x {n}  |  Total pixels: {m*n}")


U, S, Vt = np.linalg.svd(img, full_matrices=False)

def reconstruct(U, S, Vt, k):
    return np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 0, 1)

def psnr(original, approx):
    mse = np.mean((original - approx) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

def compression_ratio(m, n, k):
    return (m * n) / (k * (m + n + 1))

cumulative_energy = np.cumsum(S**2) / np.sum(S**2) * 100
k90 = int(np.argmax(cumulative_energy >= 90) + 1)
k99 = int(np.argmax(cumulative_energy >= 99) + 1)
print(f"90% energy at k = {k90}")
print(f"99% energy at k = {k99}")

ranks = [1, 5, 10, 20, 50, 100]
print(f"\n{'Rank k':>8} | {'PSNR (dB)':>10} | {'Comp. Ratio':>12} | {'Storage %':>10}")
print("-" * 52)
for k in ranks:
    Ak = reconstruct(U, S, Vt, k)
    p  = psnr(img, Ak)
    cr = compression_ratio(m, n, k)
    st = k * (m + n + 1) / (m * n) * 100
    print(f"{k:>8} | {p:>10.2f} | {cr:>12.2f} | {st:>9.1f}%")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Figure 1: SVD Image Compression — Rank Approximations',
             fontsize=13, fontweight='bold')

axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

for idx, k in enumerate(ranks):
    row, col = (idx + 1) // 4, (idx + 1) % 4
    Ak = reconstruct(U, S, Vt, k)
    p  = psnr(img, Ak)
    cr = compression_ratio(m, n, k)
    axes[row, col].imshow(Ak, cmap='gray', vmin=0, vmax=1)
    axes[row, col].set_title(f'Rank k = {k}\nPSNR = {p:.1f} dB\nCR = {cr:.1f}x',
                              fontsize=9)
    axes[row, col].axis('off')

axes[1, 3].axis('off')
plt.tight_layout()
plt.savefig('figure1_rank_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure1_rank_comparison.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 2: Singular Value Analysis', fontsize=13, fontweight='bold')

ax1.semilogy(S, color='steelblue', linewidth=2)
ax1.axvline(x=k90, color='green', linestyle='--', linewidth=1.5,
            label=f'k={k90} (90% energy)')
ax1.axvline(x=k99, color='red', linestyle='--', linewidth=1.5,
            label=f'k={k99} (99% energy)')
ax1.set_xlabel('Index i', fontsize=11)
ax1.set_ylabel('Singular Value σᵢ  (log scale)', fontsize=11)
ax1.set_title('Singular Value Decay')
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2.plot(cumulative_energy, color='darkorange', linewidth=2)
ax2.axhline(y=90, color='green', linestyle='--', linewidth=1.5,
            label='90% energy')
ax2.axhline(y=99, color='red', linestyle='--', linewidth=1.5,
            label='99% energy')
ax2.axvline(x=k90, color='green', linestyle=':', alpha=0.8)
ax2.axvline(x=k99, color='red', linestyle=':', alpha=0.8)
ax2.set_xlabel('Number of Singular Values k', fontsize=11)
ax2.set_ylabel('Cumulative Energy (%)', fontsize=11)
ax2.set_title(f'Cumulative Energy  (90% @ k={k90},  99% @ k={k99})')
ax2.set_ylim([0, 101])
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('figure2_singular_values.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure2_singular_values.png")

k_range = list(range(1, min(150, m), 2))
psnrs, crs = [], []
for k in k_range:
    Ak = reconstruct(U, S, Vt, k)
    psnrs.append(psnr(img, Ak))
    crs.append(compression_ratio(m, n, k))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 3: Quality–Compression Trade-off', fontsize=13, fontweight='bold')

ax1.plot(k_range, psnrs, color='steelblue', linewidth=2)
ax1.axhline(y=30, color='green', linestyle='--', linewidth=1.5,
            label='30 dB — Good quality')
ax1.axhline(y=40, color='orange', linestyle='--', linewidth=1.5,
            label='40 dB — Excellent quality')
ax1.set_xlabel('Rank k', fontsize=11)
ax1.set_ylabel('PSNR (dB)', fontsize=11)
ax1.set_title('PSNR vs. Rank k')
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2.plot(k_range, crs, color='coral', linewidth=2)
ax2.set_xlabel('Rank k', fontsize=11)
ax2.set_ylabel('Compression Ratio', fontsize=11)
ax2.set_title('Compression Ratio vs. Rank k')
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('figure3_psnr_vs_rank.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: figure3_psnr_vs_rank.png")

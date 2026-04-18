import matplotlib.pyplot as plt
import numpy as np

def plot_performance():
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    
    # Data dari laporanmu
    # Semua 1.0, kecuali M
    f1_scores = [1.0] * 26
    f1_scores[12] = 0.89  # Huruf M (urutan ke-13, index 12)
    
    recalls = [1.0] * 26
    recalls[12] = 0.80    # Huruf M Recall
    
    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Kita plot F1-Score saja karena Precision rata 1.0 (membosankan kalau diplot)
    # Warna biru untuk nilai sempurna, Merah untuk yang drop (M)
    colors = ['#1f77b4' if s == 1.0 else '#d62728' for s in f1_scores]
    
    bars = ax.bar(x, f1_scores, width=0.6, color=colors, alpha=0.8)

    # Tambahkan garis batas 1.0
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    # Label dan Judul
    ax.set_ylabel('F1-Score')
    ax.set_title('Performa Deteksi per Kelas Huruf BISINDO (F1-Score)', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.1) # Biar ada ruang di atas

    # Tambahkan label angka di atas batang
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        text = "1.0" if score == 1.0 else f"{score:.2f}"
        font_weight = 'bold' if score < 1.0 else 'normal'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                text,
                ha='center', va='bottom', fontsize=10, fontweight=font_weight)

    # Tambahkan kotak penjelasan untuk M
    ax.annotate('Satu sampel M\ngagal terdeteksi\n(Recall 0.80)',
                xy=(12, 0.89), xytext=(12, 0.6),
                arrowprops=dict(facecolor='black', shrink=0.05),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1),
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('chart_f1_score.png', dpi=300)

plot_performance()
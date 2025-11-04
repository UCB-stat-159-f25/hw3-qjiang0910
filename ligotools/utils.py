import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)
    hf = np.fft.rfft(strain)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


def write_wavfile(filename, fs, data):
    d = np.int16(data / np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(str(filename), int(fs), d)


def reqshift(data, fshift=100, sample_rate=4096):
    """Frequency shift the signal by constant."""
    x = np.fft.rfft(data)
    T = len(data) / float(sample_rate)
    df = 1.0 / T
    nbins = int(fshift / df)
    y = np.roll(x.real, nbins) + 1j * np.roll(x.imag, nbins)
    y[0:nbins] = 0.0
    z = np.fft.irfft(y)
    return z

def plot_asd_and_template(
    datafreq,
    template_fft,
    d_eff,
    freqs,
    data_psd,
    det,
    fs,
    outpath,
    pcolor="C0",
    title_prefix="ASD and template around event",
):
    """
    Recreates the ASD+template plot previously in the notebook's PSD cell
    and saves it to `outpath`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    template_f = np.abs(template_fft) * np.sqrt(np.abs(datafreq)) / d_eff
    ax.loglog(datafreq, template_f, "k", label="template(f)*sqrt(f)")
    ax.loglog(freqs, np.sqrt(data_psd), pcolor, label=f"{det} ASD")

    ax.set_xlim(20, fs / 2)
    ax.set_ylim(1e-24, 1e-20)
    ax.grid()
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("strain noise ASD (strain/rtHz), template h(f)*rt(f)")
    ax.legend(loc="upper left")
    ax.set_title(f"{det} {title_prefix}")

    fig.savefig(str(outpath), dpi=150, bbox_inches="tight")
    return fig, ax
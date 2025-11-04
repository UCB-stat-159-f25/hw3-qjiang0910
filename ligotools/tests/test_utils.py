from pathlib import Path
import numpy as np
from scipy.io import wavfile
from ligotools.utils import reqshift, write_wavfile

def test_reqshift_moves_peak_frequency():
    # synthetic 200 Hz sine, shift by +100 Hz -> expect ~300 Hz peak
    fs = 4096
    T = 1.0
    t = np.arange(int(fs*T)) / fs
    f0 = 200
    x = np.sin(2*np.pi*f0*t)

    y = reqshift(x, fshift=100, sample_rate=fs)

    # find dominant frequency by FFT argmax
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1/fs)
    f_peak = freqs[np.argmax(np.abs(Y))]

    assert abs(f_peak - 300) < 5, f"Expected ~300 Hz, got {f_peak:.2f} Hz"

def test_write_wavfile_roundtrip(tmp_path: Path):
    fs = 4096
    # some float data inside [-1, 1]
    t = np.linspace(0, 1, fs, endpoint=False)
    x = 0.8*np.sin(2*np.pi*220*t)
    out = tmp_path / "test.wav"

    write_wavfile(out, fs, x)

    r_fs, data = wavfile.read(out)
    # sample rate preserved
    assert r_fs == fs
    # data written as int16 with correct length
    assert data.dtype == np.int16
    assert len(data) == len(x)
    # values within full-scale int16 range
    assert data.min() >= -32768 and data.max() <= 32767
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests


def load_remote_image(url: str) -> np.ndarray:
    """Pobiera zdjęcie ze zdalnego adresu URL (np.

    Wikimedia Commons) i konwertuje do formatu OpenCV (BGR).
    """
    headers = {
        # Wikimedia wymusza podanie nagłówka User-Agent dla zapytań HTTP
        "User-Agent": "HistogramAnalysisBot/1.0 (contact@example.com)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Błąd podczas pobierania obrazu: {e}")

    # Dekodowanie bajtów obrazu do tablicy NumPy, a następnie do OpenCV BGR
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(
            "Nie udało się zdekodować obrazu. Sprawdź format pliku."
        )

    return img


def analyze_histogram_quality(img_bgr: np.ndarray) -> dict:
    """Analizuje histogram obrazu i ocenia jego cechy techniczne:

    - Ekspozycję (niedoświetlenie / prześwietlenie)
    - Kontrast (odchylenie standardowe / rozpiętość tonalna)
    - Uttratę szczegółów w cieniach i światłach (Clipping)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size

    # Statystyki jasności
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Obliczenie histogramu dla odcieni szarości (256 poziomów)
    hist_gray, _ = np.histogram(gray, bins=256, range=(0, 256))

    # Analiza obcięć tonalnych (clipping) w najniższych (0-5) i najwyższych (250-255) wartościach
    shadow_clipping = np.sum(hist_gray[:6]) / total_pixels * 100
    highlight_clipping = np.sum(hist_gray[250:]) / total_pixels * 100

    # Ocena ekspozycji
    if mean_val < 85:
        exposure_status = "Zdjęcie niedoświetlone (Underexposed)"
    elif mean_val > 170:
        exposure_status = "Zdjęcie prześwietlone (Overexposed)"
    else:
        exposure_status = "Ekspozycja prawidłowa (Poprawnie naświetlone)"

    # Ocena kontrastu
    if std_val < 40:
        contrast_status = "Niski kontrast (Wąski zakres tonalny)"
    elif std_val > 75:
        contrast_status = "Wysoki kontrast (Szeroki zakres tonalny / Twarde światło)"
    else:
        contrast_status = "Umiarkowany / Prawidłowy kontrast"

    # Wskazówki i uwagi dotyczące jakości
    warnings = []
    if shadow_clipping > 5.0:
        warnings.append(
            f"Wysokie obcięcie w cieniach ({shadow_clipping:.1f}% pikseli jest niedoświetlonych)."
        )
    if highlight_clipping > 5.0:
        warnings.append(
            f"Wysokie obcięcie w światłach ({highlight_clipping:.1f}% pikseli jest prześwietlonych)."
        )

    if not warnings:
        warnings.append("Brak znaczącej utraty szczegółów w cieniach/światłach.")

    return {
        "mean_brightness": mean_val,
        "std_contrast": std_val,
        "exposure": exposure_status,
        "contrast": contrast_status,
        "shadow_clipping_pct": shadow_clipping,
        "highlight_clipping_pct": highlight_clipping,
        "warnings": warnings,
    }


def plot_image_and_histograms(img_bgr: np.ndarray, analysis: dict):
    """Wyświetla zdjęcie oraz histogramy (całościowy i kanały R, G, B)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Analiza Histogramu i Jakości Zdjęcia", fontsize=16)

    # 1. Oryginalny obraz
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Oryginalne Zdjęcie")
    axes[0, 0].axis("off")

    # 2. Histogram całościowy (Jasność / Szarość)
    axes[0, 1].hist(
        gray.ravel(), bins=256, range=[0, 256], color="gray", alpha=0.7
    )
    axes[0, 1].set_title("Histogram Jasności (Luminancja)")
    axes[0, 1].set_xlim([0, 256])

    # 3. Informacje tekstowe z oceny jakości
    axes[0, 2].axis("off")
    report_text = (
        f"--- ROCENA JAKOŚCI ---\n\n"
        f"Średnia jasność: {analysis['mean_brightness']:.2f}\n"
        f"Odchylenie std (kontrast): {analysis['std_contrast']:.2f}\n\n"
        f"Ekspozycja: {analysis['exposure']}\n"
        f"Kontrast: {analysis['contrast']}\n\n"
        f"Uwagi:\n" + "\n".join(f"- {w}" for w in analysis["warnings"])
    )
    axes[0, 2].text(
        0.0,
        0.9,
        report_text,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )

    # 4-6. Histogramy dla poszczególnych kanałów R, G, B
    colors = ("b", "g", "r")
    titles = ("Kanał Niebieski (Blue)", "Kanał Zielony (Green)", "Kanał Czerwony (Red)")

    for i, (col, title) in enumerate(zip(colors, titles)):
        ax = axes[1, i]
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Przykładowe sprawdzone zdjęcie ze zbiorów Wikimedia Commons
    IMAGE_URL = "https://thumb.wikimedia.org/wikipedia/commons/thumb/1/14/Norderney%2C_D%C3%BCnenlandschaft%2C_Wegweiser_--_2025_--_9437.jpg/1920px-Norderney%2C_D%C3%BCnenlandschaft%2C_Wegweiser_--_2025_--_9437.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail"

    print("Pobieranie zdjęcia ze zdalnego serwera...")
    img = load_remote_image(IMAGE_URL)

    print("Analizowanie właściwości histogramu...")
    quality_report = analyze_histogram_quality(img)

    print("\nWygenerowany raport jakości:")
    for key, val in quality_report.items():
        print(f"{key}: {val}")

    print("\nGenerowanie wykresów...")
    plot_image_and_histograms(img, quality_report)
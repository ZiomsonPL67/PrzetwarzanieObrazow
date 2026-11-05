import urllib.request

import cv2
import matplotlib.pyplot as plt
import numpy as np

SOURCE = "https://thumb.wikimedia.org/wikipedia/commons/thumb/1/14/Norderney%2C_D%C3%BCnenlandschaft%2C_Wegweiser_--_2025_--_9437.jpg/1920px-Norderney%2C_D%C3%BCnenlandschaft%2C_Wegweiser_--_2025_--_9437.jpg?utm_source=commons.wikimedia.org&utm_campaign=imageinfo&utm_content=thumbnail"


# 1. Wczytanie obrazu (zdalnego lub lokalnego)
def wczytaj_obraz(source: str) -> np.ndarray:
    """Zwraca obraz jako tablicę NumPy w układzie RGB."""
    if source.lower().startswith(("http://", "https://")):
        # Wikimedia odrzuca zapytania bez nagłówka User-Agent (błąd 403)
        req = urllib.request.Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            dane = np.asarray(bytearray(response.read()), dtype=np.uint8)
        obraz_bgr = cv2.imdecode(dane, cv2.IMREAD_COLOR)
    else:
        obraz_bgr = cv2.imread(source, cv2.IMREAD_COLOR)

    if obraz_bgr is None:
        raise ValueError("Nie udało się wczytać obrazu - sprawdź adres lub ścieżkę.")
    return cv2.cvtColor(obraz_bgr, cv2.COLOR_BGR2RGB)


obraz = wczytaj_obraz(SOURCE)
print(f"Obraz oryginalny: kształt = {obraz.shape}, typ = {obraz.dtype}")

# 2. Wyświetlenie obrazu oryginalnego
plt.figure(figsize=(6, 6))
plt.imshow(obraz)
plt.title(f"Obraz oryginalny {obraz.shape[1]}x{obraz.shape[0]} px")
plt.axis("off")
plt.show()

# 3. Zmniejszenie rozdzielczości o 50% i konwersja do skali szarości
zmniejszony = cv2.resize(obraz, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
szary = cv2.cvtColor(zmniejszony, cv2.COLOR_RGB2GRAY)
print(f"Po zmniejszeniu i konwersji do grayscale: kształt = {szary.shape}")

# 4. Obrót o 90 stopni (zgodnie z ruchem wskazówek zegara)
obrocony = cv2.rotate(szary, cv2.ROTATE_90_CLOCKWISE)
# równoważnie w NumPy (przeciwnie do ruchu wskazówek): np.rot90(szary)
print(f"Po obrocie: kształt = {obrocony.shape}")

# 5. Wyświetlenie obrazu wynikowego
plt.figure(figsize=(6, 6))
plt.imshow(obrocony, cmap="gray", vmin=0, vmax=255)
plt.title(f"Obraz wynikowy {obrocony.shape[1]}x{obrocony.shape[0]} px (grayscale, obrót 90°)")
plt.axis("off")
plt.show()

# 6. Macierz obrazu w postaci tablicy liczb
print("\nMacierz obrazu wynikowego (NumPy skraca duże tablice):")
print(obrocony)

print("\nFragment 10x10 (lewy górny róg):")
with np.printoptions(linewidth=200):
    print(obrocony[:10, :10])

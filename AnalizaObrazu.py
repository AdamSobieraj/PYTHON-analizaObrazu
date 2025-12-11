import cv2
import os
import glob

# KONFIGURACJA ALGORYTMU (Tuning)
# 1. PARAMETRY OBRAZU I PODGLĄDU
INPUT_FOLDER = 'zdjecia_wejsciowe'
OUTPUT_FOLDER = 'wyniki_kodow'
PREVIEW_WIDTH = 400  # Szerokość okienka podglądu (w pikselach)

# 2. PARAMETRY PROGOWANIA (THRESHOLD)
# Określa, jak bardzo "czarne" muszą być paski, by zostały wykryte.
# Zakres 0-255. Wyższa wartość (np. 230) wykryje tylko idealny kontrast.
# Niższa (np. 180) wykryje też kody w cieniu, ale może łapać szum.
THRESH_VALUE = 225

# 3. PARAMETRY ROZMYCIA (BLUR)
# Pomaga usunąć szum przed binaryzacją.
# Musi być liczbą nieparzystą, np. (9, 9), (5, 5).
BLUR_KERNEL_SIZE = (9, 9)

# 4. PARAMETRY MORFOLOGII (ZLEWANIE PASKÓW)
# Kształt prostokąta używany do łączenia pionowych pasków w jedną plamę.
# (Szerokość, Wysokość). Musi być szeroki, żeby łączyć paski w poziomie.
MORPH_KERNEL_SIZE = (21, 7)

# Liczba powtórzeń czyszczenia (erozji/dylatacji).
# Więcej iteracji = gładsze bloki, ale mniejsze kody mogą zniknąć.
ITERATIONS = 4

# 5. FILTROWANIE WYNIKÓW
# Minimalna powierzchnia (w pikselach), aby uznać obiekt za kod.
# Zmniejsz, jeśli kody są daleko/małe. Zwiększ, jeśli wykrywa śmieci.
MIN_AREA = 2000

# Minimalna proporcja (Szerokość / Wysokość).
# Kody są zazwyczaj szersze niż wyższe. Wartość 1.5 oznacza, że
# szerokość musi być min. 1.5x większa od wysokości.
MIN_ASPECT_RATIO = 1.5

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def show_step(title, image, x_pos=0, y_pos=0):

    # Wyświetla obraz w przeskalowanym oknie
    h, w = image.shape[:2]

    if w > PREVIEW_WIDTH:
        scale = PREVIEW_WIDTH / w
        dim = (int(w * scale), int(h * scale))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = image

    cv2.imshow(title, resized)
    cv2.moveWindow(title, x_pos, y_pos)


def process_and_show(filepath, filename):
    print(f"Przetwarzanie: {filename}...")

    image = cv2.imread(filepath)
    if image is None:
        print("Błąd pliku.")
        return True

        # --- ETAP 1: Oryginał ---
    show_step("1. Oryginal", image, 0, 0)

    # --- ETAP 2: Skala szarości ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_step("2. Szarosc", gray, PREVIEW_WIDTH + 10, 0)

    # --- ETAP 3: Gradient (Sobel X - Y) ---
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    show_step("3. Gradient", gradient, (PREVIEW_WIDTH * 2) + 20, 0)

    # --- ETAP 4: Binaryzacja ---
    blurred = cv2.blur(gradient, BLUR_KERNEL_SIZE)
    (_, thresh) = cv2.threshold(blurred, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    show_step("4. Binaryzacja", thresh, 0, 350)

    # --- ETAP 5: Morfologia ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=ITERATIONS)
    closed = cv2.dilate(closed, None, iterations=ITERATIONS)
    show_step("5. Morfologia", closed, PREVIEW_WIDTH + 10, 350)

    # --- ETAP 6: Wynik (Kontury) ---
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    count = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h

        # Użycie parametrów konfiguracyjnych do filtrowania
        if area > MIN_AREA and ar > MIN_ASPECT_RATIO:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            count += 1

    show_step("6. Wynik", output_image, (PREVIEW_WIDTH * 2) + 20, 350)

    # Zapis wyniku na dysk
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, "processed_" + filename), output_image)
    print(f" -> Znaleziono kodów: {count}. (SPACJA - dalej, Q - wyjście)")

    # Oczekiwanie na klawisz
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == ord('q'):
        return False

    return True


def main():
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not files:
        print(f"Brak zdjęć w folderze {INPUT_FOLDER}")
        return

    for file_path in files:
        filename = os.path.basename(file_path)
        if not process_and_show(file_path, filename):
            break

    print("Koniec pracy.")


if __name__ == "__main__":
    main()
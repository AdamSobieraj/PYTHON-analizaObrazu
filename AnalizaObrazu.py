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
# [POPRAWKA]: Używamy flagi cv2.THRESH_OTSU, która ignoruje tę wartość
# i dobiera idealny próg dynamicznie.
THRESH_VALUE = 200

# 3. PARAMETRY ROZMYCIA (BLUR)
# Pomaga usunąć szum przed binaryzacją.
# Musi być liczbą nieparzystą, np. (9, 9), (5, 5).
BLUR_KERNEL_SIZE = (3, 3)

# 4. PARAMETRY MORFOLOGII (ZLEWANIE PASKÓW)
# Kształt prostokąta używany do łączenia pionowych pasków w jedną plamę.
# (Szerokość, Wysokość). Musi być szeroki, żeby łączyć paski w poziomie.
MORPH_KERNEL_SIZE = (9, 3)

# [NOWA OPCJA] Wybór metody morfologicznej:
# True  = ZAMYKANIE (CLOSE) -> Dylatacja potem Erozja. Zalecane do łączenia pasków kodu w jeden blok.
# False = OTWIERANIE (OPEN) -> Erozja potem Dylatacja. Służy do usuwania szumu (kropek), ale może "pociąć" kod.
USE_MORPH_CLOSE = True

# Liczba powtórzeń czyszczenia (erozji/dylatacji).
# Więcej iteracji = gładsze bloki, ale mniejsze kody mogą zniknąć.
ITERATIONS = 1

# 5. FILTROWANIE WYNIKÓW
# Minimalna powierzchnia (w pikselach), aby uznać obiekt za kod.
# Zmniejsz, jeśli kody są daleko/małe. Zwiększ, jeśli wykrywa śmieci.
MIN_AREA = 1500

# Minimalna proporcja (Szerokość / Wysokość).
# Kody są zazwyczaj szersze niż wyższe. Wartość 1.5 oznacza, że
# szerokość musi być min. 1.5x większa od wysokości.
MIN_ASPECT_RATIO = 2.5

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

    # --- ETAP 2: HSV -> Kanał Value ---
    # Konwertujemy BGR na HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Rozdzielamy kanały: H (Barwa), S (Nasycenie), V (Jasność)
    h, s, v = cv2.split(hsv)
    # Używamy tylko kanału V (Value) jako naszej "szarości".
    gray = v
    show_step("2. HSV - Kanal V (Jasnosc)", gray, PREVIEW_WIDTH + 10, 0)

    # --- ETAP 3: Gradient (Sobel X - Y) ---
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradient = cv2.convertScaleAbs(gradX)
    show_step("3. Gradient", gradient, (PREVIEW_WIDTH * 2) + 20, 0)

    # --- ETAP 4: Binaryzacja ---
    blurred = cv2.blur(gradient, BLUR_KERNEL_SIZE)

    # [POPRAWKA] Używamy OTSU dla automatycznego progu
    (_, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    show_step("4. Binaryzacja (Otsu)", thresh, 0, 350)

    # --- ETAP 5: Morfologia ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)

    # [NOWA LOGIKA] Wybór metody na podstawie zmiennej USE_MORPH_CLOSE
    if USE_MORPH_CLOSE:
        # ZAMYKANIE (CLOSE): Wypełnia dziury między paskami
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        method_name = "CLOSE"
    else:
        # OTWIERANIE (OPEN): Usuwa małe śmieci z tła
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        method_name = "OPEN"

    # Dodatkowe czyszczenie (Erozja i Dylatacja)
    morphed = cv2.erode(morphed, None, iterations=ITERATIONS)
    morphed = cv2.dilate(morphed, None, iterations=ITERATIONS)

    data = morphed.copy()
    show_step(f"5. Morfologia ({method_name})", morphed, PREVIEW_WIDTH + 10, 350)

    # --- ETAP 6: Wynik (Kontury) ---
    cnts, _ = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sortowanie konturów od największego
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    output_image = image.copy()
    count = 0

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h

        # Użycie parametrów konfiguracyjnych do filtrowania
        if area > MIN_AREA and ar > MIN_ASPECT_RATIO:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Dodatek: wyświetlanie pola powierzchni
            cv2.putText(output_image, f"Area: {int(area)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            count += 1

            # break # Odkomentuj, jeśli chcesz tylko 1 (największy) kod

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
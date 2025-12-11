# Wykrywacz Kodów Kreskowych (Python + OpenCV)

Prosty program w języku Python służący do identyfikacji lokalizacji kodów kreskowych na zdjęciach (np. szafy serwerowe, magazyny).

Rozwiązanie **nie wykorzystuje sztucznej inteligencji** (sieci neuronowych). Działa w oparciu o klasyczne metody przetwarzania obrazu (Computer Vision): analizę gradientów, operacje morfologiczne i wykrywanie konturów.

## Wymagania

Aby uruchomić program, musisz mieć zainstalowany na komputerze:
1.  **Python** (wersja 3.x)
2.  Biblioteki zewnętrzne wymienione poniżej.

## Instalacja

Otwórz terminal (konsolę) i wpisz poniższe komendy, aby zainstalować wymagane biblioteki:

```bash
pip install opencv-python
pip install numpy
```

> **Uwaga:** W kodzie używamy `import cv2`, ale paczka do instalacji nazywa się `opencv-python`.

## Struktura projektu

Przed uruchomieniem upewnij się, że Twoje pliki są ułożone w następujący sposób:

```text
Twój_Projekt/
│
├── main.py                # Główny plik z kodem programu
├── zdjecia_wejsciowe/     # (Folder) Tu wrzuć swoje zdjęcia (.jpg, .png)
└── wyniki_kodow/          # (Folder) Tutaj program zapisze przetworzone zdjęcia
                           # (Ten folder utworzy się sam po uruchomieniu)
```

## Obsługa programu

Program działa w trybie wsadowym z podglądem krok po kroku:

1.  Program wczyta pierwsze zdjęcie z folderu `zdjecia_wejsciowe`.
2.  Wyświetli **6 okien** pokazujących każdy etap przetwarzania (Oryginał, Szarość, Gradient, Binaryzacja, Morfologia, Wynik).
3.  **Naciśnij SPACJĘ** (lub dowolny inny klawisz), aby zamknąć okna bieżącego zdjęcia i przejść do następnego.
4.  **Naciśnij klawisz `q`**, jeśli chcesz przerwać działanie programu przed przetworzeniem wszystkich zdjęć.

Przetworzone zdjęcia z zaznaczonymi kodami (zielone ramki) są automatycznie zapisywane w folderze `wyniki_kodow`.

## Jak to działa (Algorytm)

Program wykonuje następujące operacje na każdym zdjęciu:
1.  **Grayscale:** Konwersja na odcienie szarości.
2.  **Sobel Gradient:** Wykrywanie obszarów o dużym kontraście poziomym (pionowe paski kodu) i usuwanie linii poziomych (np. półek).
3.  **Blur & Threshold:** Rozmycie i zamiana obrazu na czarno-biały, aby wyodrębnić silne krawędzie.
4.  **Morphology:** Zastosowanie "zamknięcia" (Closing) szerokim prostokątnym jądrem. Powoduje to zlanie się pasków kodu kreskowego w jeden lity prostokąt.
5.  **Contours:** Znalezienie obrysów białych plam i filtrowanie ich po wielkości oraz proporcjach (szukamy obiektów prostokątnych).

## Podsumowanie

Dlaczego nie użyto YOLO (AI)?

W tym projekcie celowo zrezygnowano z modelu **YOLO** (You Only Look Once) i metod opartych na sieciach neuronowych (Deep Learning) na rzecz **klasycznego przetwarzania obrazu** (OpenCV). Decyzja ta wynika z następujących przesłanek:

1.  **Brak konieczności trenowania (Dataset):**
    *   Modele AI (jak YOLO) wymagają zbioru tysięcy opisanych ręcznie zdjęć, aby "nauczyć się", czym jest kod kreskowy.
    *   Metoda klasyczna wykorzystuje fakt, że kody kreskowe mają unikalne cechy geometryczne (seria pionowych linii o wysokim kontraście), które można opisać matematycznie bez żadnego procesu uczenia.

2.  **Wydajność i zasoby (CPU vs GPU):**
    *   YOLO jest ciężkim modelem obliczeniowym i do szybkiego działania zazwyczaj wymaga dedykowanej karty graficznej (GPU - CUDA).
    *   Użyty tutaj algorytm oparty na gradientach jest bardzo lekki, działa błyskawicznie na każdym standardowym procesorze (CPU) i zużywa minimalną ilość pamięci RAM.

3.  **Prostota wdrożenia:**
    *   Uruchomienie YOLO wymaga instalacji ciężkich frameworków (PyTorch, TensorFlow) i zajmuje dużo miejsca na dysku (często >1GB).
    *   Nasze rozwiązanie wymaga jedynie lekkiej biblioteki `opencv-python` i `numpy`.

4.  **"Strzelanie z armaty do muchy":**
    *   Kody kreskowe są obiektami syntetycznymi, stworzonymi tak, aby były łatwe do wykrycia przez maszyny. Używanie zaawansowanej sztucznej inteligencji do wykrywania prostych wzorów geometrycznych jest często nadmiarowe (over-engineering).
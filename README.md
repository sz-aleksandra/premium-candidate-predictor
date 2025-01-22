# IUM - projekt 24Z
## Oryginalna treść zadania
**Zadanie 10:**  
“Jakiś czas temu wprowadziliśmy konta premium, które uwalniają użytkowników od słuchania reklam. Nie są one jednak jeszcze zbyt popularne – czy możemy się dowiedzieć, które osoby są bardziej skłonne do zakupu takiego konta?”
## Autorzy
- Gustaw Malinowski
- Aleksandra Szymańska
- Kacper Straszak

## Instalacja wymaganych bibliotek
```
pip install tqdm scikit-learn numpy python-dateutil flask pandas matplotlib requests
```

## Uruchamianie
1. `microservice.py` - uruchomienie mikroserwisu
2. `get_microservice_responses.py` - test mikroserwisu

### Dokumentacja
1. `Raport z eksperymentów AB.ipynb`
2. `Raport z budowy modeli.ipynb`

### Stworzenie Modeli
- `model_creation.py`

### Stworzenie zbiorów danych
1. `modeling_dataset_creation.py` - tworzenie zbiorów danych
2. `split_data_to_train_validate_test.py` - podział na zbiory treningowy/walidacyjny/testowy 

### Pomocnicze
3. `choose_model_attributes.py` - wybór atrybutów na podstawie wag znormalizowanego modelu

## Zmiany w wersji 2
- Dodanie do wszystkich generatorów losowych ziaren
- Naprawienie Skalara tak aby nie wysyłał ostrzeżeń
- Dodanie requirements.txt
- Dodanie analizy modeli pod względem wstępnych założeń z dokumentacji wstępnej w raporcie z budowy modeli
- Przerzucenie funkcji pomocniczych z raportów oddzielnych plików
- Obsługa mikroserwisem zapytań losowo wybranym modelem (z ziarnem aby było powtarzalne)
- Dopasowywanie próbek z logów do ground truth data
- Dodanie wczytywania danych z predykcji z logów do testów AB
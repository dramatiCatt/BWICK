if __name__ == "__main__":

    import os
    import json
    import random
    from collections import defaultdict

    def load_data_paths(data_dir: str) -> list[dict[str, str | int]]:
        """
        Wczytuje ścieżki do obrazów i odpowiadających im plików JSON,
        oraz parsuje dane z JSON. Dodatkowo ekstrahuje numer palca z nazwy pliku.

        Args:
            data_dir (str): Ścieżka do katalogu głównego zawierającego obrazy i pliki JSON.

        Returns:
            List[Dict]: Lista słowników, gdzie każdy słownik reprezentuje jedną próbkę danych
                        i zawiera ścieżkę do obrazu, współrzędne core i delta (jeśli istnieje),
                        flagę is_delta_present oraz numer palca.
        """
        data_samples : list[dict[str, str | int]] = []
        
        # Przejdź przez wszystkie pliki w katalogu
        for filename in os.listdir(data_dir):
            if not filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg', '.bmp')):
                continue

            image_path: str = os.path.join(data_dir, filename)
            json_filename: str = os.path.splitext(filename)[0] + '.json'
            json_path: str = os.path.join(data_dir, json_filename)

            # Ekstrakcja numeru palca z nazwy pliku 1nn_n.tif
            try:
                # Zakładamy format 1nn_n.tif, np. 101_1.tif, 102_5.tif
                finger_id_str: str = filename.split('_')[0]
                finger_id = int(finger_id_str)
            except (IndexError, ValueError):
                print(f"Ostrzeżenie: Nie udało się wyekstrahować numeru palca z nazwy pliku: {filename}. Pomijam.")
                continue

            if not os.path.exists(json_path):
                print(f"Ostrzeżenie: Nie znaleziono pliku JSON dla obrazu: {image_path}. Pomijam.")
                continue

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    annotation: dict[str, list[int] ] = json.load(f)

                core_coords: list[int] | None = annotation.get("core")

                if core_coords is None:
                    print(f"Ostrzeżenie: Plik JSON {json_path} nie zawiera 'core'. Pomijam.")
                    continue

                data : dict[str, str | int] = {
                    "finger_id" : finger_id,
                    "image_path" : image_path,
                    "json_path" : json_path
                }

                data_samples.append(data)

            except json.JSONDecodeError:
                print(f"Błąd parsowania JSON w pliku: {json_path}. Pomijam.")
            except KeyError as e:
                print(f"Błąd klucza w pliku JSON {json_path}: {e}. Upewnij się, że 'x' i 'y' są obecne. Pomijam.")
            except Exception as e:
                print(f"Nieoczekiwany błąd podczas przetwarzania {json_path}: {e}. Pomijam.")

        print(f"Wczytano {len(data_samples)} poprawnych próbek danych.")
        return data_samples

    def split_data_by_finger(
        data_samples: list[dict[str, str | int]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        seed: int | None = 42
    ) -> dict[str, list[dict[str, str | int]]]:
        """
        Dzieli próbki danych na zbiory treningowy, walidacyjny i testowy,
        zapewniając, że każdy palec jest reprezentowany w zbiorach walidacyjnych i testowych.

        Args:
            data_samples (List[Dict]): Lista słowników próbek danych, zawierająca 'finger_id'.
            train_ratio (float): Proporcja danych treningowych.
            val_ratio (float): Proporcja danych walidacyjnych.
            test_ratio (float): Proporcja danych testowych.
            seed (Optional[int]): Ziarno dla generatora liczb losowych dla powtarzalności.

        Returns:
            Dict[str, List[Dict]]: Słownik zawierający listy próbek dla każdego zbioru ('train', 'val', 'test').
        """
        if not (train_ratio + val_ratio + test_ratio == 1.0):
            raise ValueError("Sum of train_ratio, val_ratio, and test_ratio must be 1.0")

        if seed is not None:
            random.seed(seed)

        # 1. Pogrupuj dane według numeru palca
        data_by_finger : defaultdict[int, list[dict[str, str | int,]]] = defaultdict(list[dict[str, str | int]])
        for sample in data_samples:
            finger_id: str | int = sample["finger_id"]
            assert isinstance(finger_id, int)
            data_by_finger[finger_id].append(sample)

        train_data : list[dict[str, str | int]] = []
        val_data : list[dict[str, str | int]] = []
        test_data : list[dict[str, str | int]] = []

        # 2. Dla każdego palca, rozdziel próbki do zbiorów
        for finger_id, samples_for_finger in data_by_finger.items():
            random.shuffle(samples_for_finger) # Mieszaj próbki dla danego palca

            num_samples: int = len(samples_for_finger)

            if num_samples < 2:
                print(f"Ostrzeżenie: Palec {finger_id} ma mniej niż 2 próbki ({num_samples}). Nie można zagwarantować reprezentacji we wszystkich zbiorach.")
                train_data.extend(samples_for_finger)
                continue
            
            num_train_samples: int = max(1, int(num_samples * train_ratio))
            num_val_samples: int = max(1, int(num_samples * val_ratio))
            
            train_data.extend(samples_for_finger[:num_train_samples])
            val_data.extend(samples_for_finger[num_train_samples : num_train_samples + num_val_samples])
            test_data.extend(samples_for_finger[num_train_samples + num_val_samples:])

        # Na koniec, przetasuj każdą listę, aby nie było zależności od kolejności dodawania palców
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

        print(f"Podział danych po zagwarantowaniu reprezentacji palców:")
        print(f"  Treningowe: {len(train_data)} próbek")
        print(f"  Walidacyjne: {len(val_data)} próbek")
        print(f"  Testowe: {len(test_data)} próbek")

        # Sprawdzenie unikalnych palców w każdym zbiorze
        train_fingers : set[int | str] = { s["finger_id"] for s in train_data }
        val_fingers : set[int | str] = { s["finger_id"] for s in val_data }
        test_fingers : set[int | str] = { s["finger_id"] for s in test_data }
        all_fingers: set[int] = set(data_by_finger.keys())

        print(f"  Unikalne palce w treningowym: {len(train_fingers)}/{len(all_fingers)}")
        print(f"  Unikalne palce w walidacyjnym: {len(val_fingers)}/{len(all_fingers)}")
        print(f"  Unikalne palce w testowym: {len(test_fingers)}/{len(all_fingers)}")
        if len(val_fingers) == len(all_fingers) and len(test_fingers) == len(all_fingers):
            print("  Wszystkie palce reprezentowane w zbiorach walidacyjnych i testowych!")
        else:
            print("  OSTRZEŻENIE: Nie wszystkie palce są reprezentowane w zbiorach walidacyjnych/testowych (prawdopodobnie za mało próbek dla niektórych palców).")


        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }

    def save_split_to_json(split_data: dict[str, list[dict[str, str | int]]], output_path: str):
        """
        Zapisuje podział danych (ścieżki i adnotacje) do pliku JSON.
        """
        output_data: dict[str, list[dict[str, str | int]]] = {
            "train": [{"image_path": d["image_path"], "json_path": d["json_path"]} for d in split_data["train"]],
            "val": [{"image_path": d["image_path"], "json_path": d["json_path"]} for d in split_data["val"]],
            "test": [{"image_path": d["image_path"], "json_path": d["json_path"]} for d in split_data["test"]]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Podział danych zapisany do: {output_path}")

    data_paths: list[dict[str, str | int]] = load_data_paths("./Data")

    if not data_paths:
        print("There is no data")
        exit()

    data: dict[str, list[dict[str, str | int]]] = split_data_by_finger(data_paths, 0.7, 0.1, 0.2, None)
    save_split_to_json(data, "data_split.json")
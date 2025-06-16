import tkinter as tk
from tkinter import filedialog, messagebox
import pathlib
import fingerpy as fpa
import cv2
from PIL import Image, ImageTk

last_dir = '.'

def get_img_path() -> str:
    global last_dir
    
    file_path = filedialog.askopenfilename(
        title="Wybierz plik zdjęcia",
        initialdir=last_dir,
        filetypes=(
            ("Pliki Zdjec", "*.tif *.png *.jpg *.jpeg"),
            ("Plik TIF", "*.tif"),
            ("Plik PNG", "*.png"),
            ("Plik JPG", "*.jpg"),
            ("Plik JPEG", "*.jpeg"),
            ("Wszystkie pliki", "*.*")
        )
    )
    
    if file_path:
        last_dir = pathlib.Path(file_path).parent 
    return file_path

def get_template_path() -> str:
    global last_dir
    
    file_path = filedialog.askopenfilename(
        title="Wybierz plik zdjęcia",
        initialdir=last_dir,
        filetypes=(
            ("Pliki JSON", "*.json"),
            ("Plik tekstowy", "*.txt"),
            ("Wszystkie pliki", "*.*")
        )
    )
    
    if file_path:
        last_dir = pathlib.Path(file_path).parent 
    return file_path

def get_save_file_path() -> str:
    """Otwiera okno dialogowe 'Zapisz jako...' i zwraca wybraną ścieżkę."""
    global last_dir
    
    file_path = filedialog.asksaveasfilename(
        title="Zapisz plik",
        initialdir=last_dir,
        defaultextension=".json",  # Domyślne rozszerzenie, jeśli użytkownik go nie poda
        filetypes=(
            ("Pliki JSON", "*.json"),
            ("Plik tekstowy", "*.txt"),
            ("Wszystkie pliki", "*.*")
        )
    )
    return file_path

def add_path() -> None:
    global paths_list
    
    path = get_img_path()
    if path:
        paths_list.insert(tk.END, path)
        
def clear_paths() -> None:
    global paths_list
    paths_list.delete(0, tk.END)
    
def remove_path() -> None:
    global paths_list
    
    try:
        idx = paths_list.curselection()[0]
        paths_list.delete(idx)
    except IndexError:
        messagebox.showinfo("Błąd", "Wybierz ścieżkę do usunięcia.")
    
def create_template() -> None:
    save_path = get_save_file_path()
    
    if not save_path:
        return
    
    global paths_list
    
    messagebox.showinfo("Zapisywanie", f"Zapisywanie do pliku:\n{save_path}")
    fpa.create_and_save_templates(save_path, paths_list.get(0, tk.END))
    messagebox.showinfo("Sukces", f"Wzorce został zapisane pomyślnie do:\n{save_path}")

def open_create_template_window() -> None:
    global root
    
    create_window = tk.Toplevel(root)
    create_window.title("Stwórz wzorce odcisków")
    create_window.geometry("500x350")
    
    global paths_list
    paths_list = tk.Listbox(create_window, width=60)
    paths_list.pack(pady=10)
    
    list_buttons_frame = tk.Frame(create_window)
    list_buttons_frame.pack(pady=5)
    
    add_button = tk.Button(list_buttons_frame, text="Dodaj ścieżkę", command=add_path)
    add_button.pack(side=tk.LEFT, padx=5)
    
    clear_button = tk.Button(list_buttons_frame, text="Wyczyść ścieżki", command=clear_paths)
    clear_button.pack(side=tk.LEFT, padx=5)
    
    remove_button = tk.Button(list_buttons_frame, text="Usuń ścieżkę", command=remove_path)
    remove_button.pack(side=tk.LEFT, padx=5)
    
    create_template_button = tk.Button(create_window, text="Utwórz template", command=create_template)
    create_template_button.pack(pady=10)

def get_fingerprint_img() -> None:
    path = get_img_path()
    
    global img_path_label, image_label, current_fingerprint
    
    if not path:
        img_path_label.config(text="Nie wybrano pliku.")
        image_label.config(image="")
        current_fingerprint = None
        return
    
    try:
        original_img = fpa.img.load_img(path)
        current_fingerprint = fpa.Fingerprint(original_img)
        
        current_fingerprint.show_all_steps()
        
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        
        display_img_pil = img_pil.copy()
        display_img_pil.thumbnail((400, 400))
        
        img_tk = ImageTk.PhotoImage(display_img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        img_path_label.config(text=f"Wybrane zdjęcie: {path}")
    except ValueError as ve:
        messagebox.showerror("Błąd Wczytywania", str(ve))
        image_label.config(image="")
        current_fingerprint = None
        img_path_label.config(text="Błąd ładowania zdjęcia.")
    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił nieoczekiwany błąd: {e}")
        image_label.config(image="")
        current_fingerprint = None
        img_path_label.config(text="Błąd ładowania zdjęcia.")

def get_template_file() -> None:
    path = get_template_path()
    
    global templates_data_path_label, current_templates
    
    if not path:
        templates_data_path_label.config(text="Nie wybrano pliku.")
        current_templates = None
        return
    
    try:
        current_templates = fpa.load_templates(path)
        templates_data_path_label.config(text=f"Wybrany wzorzec: {path}")
    except ValueError as ve:
        messagebox.showerror("Błąd Wczytywania", str(ve))
        current_templates = None
        templates_data_path_label.config(text="Błąd ładowania wzorca.")
    except Exception as e:
        messagebox.showerror("Błąd", f"Wystąpił nieoczekiwany błąd: {e}")
        current_templates = None
        templates_data_path_label.config(text="Błąd ładowania wzorca.")
        
def authenticate() -> None:
    global current_templates, current_fingerprint
    
    if current_templates is None or current_fingerprint is None:
        messagebox.showerror("Błąd", "Nie zaczytano albo palca albo wzorca")
        return
    
    minuiae = fpa.FingerprintTemplate.from_fingerprint(current_fingerprint).minutiae
    
    result = fpa.authenticate(minuiae, current_templates, 0.4)
    
    messagebox.showinfo("Sukcess" if result else "Porażka", f"Autentykacja sie {'' if result else 'nie '}powiodla")

# TODO: gaussian blure na znormalizowanym obrazie
# TODO: filtr medianowy na znormalizowanym obrazku
# TODO: porównywanie minucji jak w pracy Stolarka na wikampie
# TODO: może spróbować sieć neuronową dla wykrywania core i delty

def main():
    global root, current_fingerprint, img_path_label, image_label, templates_data_path_label, current_templates
    
    current_fingerprint = None
    current_templates = None
    
    root = tk.Tk()
    root.title("Autentykacja Lini Papilarnych")
    root.geometry("600x550")
    
    img_path_label = tk.Label(root, text="Nie wybrano pliku.", wraplength=550)
    img_path_label.pack(pady=10)
    
    image_label = tk.Label(root, bd=2, relief='groove')
    image_label.pack(pady=10)
    
    load_image_button = tk.Button(root, text="Wczytaj zdjęcie", command=get_fingerprint_img)
    load_image_button.pack(pady=10)
    
    templates_data_frame = tk.Frame(root)
    templates_data_frame.pack(pady=5)
    
    templates_data_path_label = tk.Label(templates_data_frame, text="Nie wybrano pliku.", wraplength=500)
    templates_data_path_label.pack(side=tk.LEFT, padx=5)
    
    load_templates_data_button = tk.Button(templates_data_frame, text="Wczytaj wzorce", command=get_template_file)
    load_templates_data_button.pack(side=tk.LEFT, padx=5)
    
    some_buttons_frame = tk.Frame(root)
    some_buttons_frame.pack(pady=5)
    
    authenticat_button = tk.Button(some_buttons_frame, text="Autentykuj", command=authenticate)
    authenticat_button.pack(side=tk.LEFT, padx=5)
    
    create_template_button = tk.Button(some_buttons_frame, text="Stwórz wzorzec", command=open_create_template_window)
    create_template_button.pack(side=tk.LEFT, padx=5)
    
    root.mainloop()

if __name__ == "__main__":
    main()
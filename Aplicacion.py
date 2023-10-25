# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 21:05:28 2023

@author: Alexander Passo 
Ingeniería Electrónica 
Universidad de Antioquia
2023-2

La siguiente aplicación tiene como objetivo demostrar los conocimientos adquiridos sobre
la transformada discreta de fourier. 
La aplicación abrirá una ventana en el cuál le pedirá que añada el audio del piano. Es importante
tener en cuenta que el audio debe ser solo PIANO para que la aplicación funcione correctamente.
Luego de añadir el audio, lo puede reproducir dandole clic en el boton "reproducir", y luego
puede obtener las gráficas en tiempo y el análisis espectral. No es necesario reproducir el audio
para obtener las gráficas. 
"""

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

import sounddevice as sd
from scipy.signal import find_peaks



#declaracion de funciones

#Funcion que me permite agregar el audio
def agregar():
    global fs, signal, x_senal1
    signal = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav *.mp3 *.ogg")])
    if signal:
        # Guardar la ubicación del archivo en una variable
        print("Archivo seleccionado:", signal)   
        
        fs, x_senal1 = read(signal)
        
        btn_agregar.config(state=tk.DISABLED)       #Desactivo el boton añadir 
        btn_reproducir.config(state=tk.NORMAL)      #Activo el boton reproducir
        btn_graficas.config(state=tk.NORMAL)        #Activo el boton graficas
        
        
#Reproduce el audio
def reproducir():
    # Agrega aquí la lógica para la función "Reproducir"
    # Carga el archivo de audio (reemplaza 'audio_file.mp3' con la ruta de tu archivo)
    # Load the audio file
    #audio_data = sd.read(signal)
    global x_senal1
    # Play the audio file
    sd.play(x_senal1)
    
    # Wait for the audio file to finish playing
    sd.wait()
    

#Para construir la matriz de DFT
def dftmatrix(N, Nfft):
    # construct DFT matrix
    k = np.arange(Nfft)
    if N is None:
        N = Nfft
    n = np.arange(N)
    U = np.matrix(np.exp(1j * 2 * np.pi / Nfft * k * n[:, None]))
    return U / np.sqrt(Nfft)


#Permite obtener la grafica en el tiempo y frecuencia
def ventanas_graficas(t,signal):
    global Xf1_abs, freq, peaks,resize_shap
    nfft = 1024
    new_signal = signal[:80000]
    new_signal.shape = (len(new_signal), 1)  # Convierto en vector columna para poder multiplicar con la matriz
    U1 = dftmatrix(len(new_signal), nfft)  # calculo la matriz de transformación
    Xf1 = U1.H * new_signal[:]  # calculo la fft
    Xf1 = Xf1 / float(np.max(np.abs(Xf1)))      #Normalizando la amplitud del espectro
    freq = np.hstack((np.arange(0, nfft // 2 - 1), np.arange(-nfft // 2, 1))) * fs / nfft
    
    Xf1_abs = np.abs(Xf1)
    #peaks = argrelmax(Xf1_abs)
    #peaks, _ = find_peaks(Xf1_abs)  # Ajusta la altura según tus necesidades
        
    resize = np.array(Xf1_abs.flatten())
    
    resize_shap = resize.reshape(1024,)
    
    peaks, _ = find_peaks(resize_shap, height=0.1)
    

    # Crear una nueva ventana de tkinter
    root = tk.Tk()
    root.title("Gráfica de Señal y Transformada de Fourier")

    # Crear una figura de Matplotlib y agregarla a la ventana de tkinter
    fig, ax = plt.subplots(2, 1, figsize=(15, 10))

    # Gráfica de la señal original
    ax[0].plot(t, signal)
    ax[0].set_xlabel('Tiempo (s)')
    ax[0].set_ylabel('Amplitud')
    ax[0].set_title('Señal Original')
    ax[0].grid()

    # Gráfica de la Transformada de Fourier
    ax[1].plot(freq, np.abs(Xf1), label="x1")
    # Marca los picos en la gráfica de la Transformada de Fourier
    ax[1].plot(freq[peaks], Xf1_abs[peaks], 'ro', label="Picos")  # 'ro' significa puntos rojos
    ax[1].legend()
    ax[1].set_xlabel('Frecuencia (Hz)')
    ax[1].set_ylabel('Amplitud')
    ax[1].set_title('Espectro ')
    ax[1].set_xlim(-2000, 2000)
    ax[1].grid()

        
    #Muestro las frecuencias halladas
    mostrar_frecuencias(peaks, freq)
    
   
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()
    
    
#Se define una nueva ventana para mostrar las frecuencias de cada pico y la Nota del piano
def mostrar_frecuencias(peaks, freq):
    # Crear una nueva ventana de tkinter
    frecuencias_window = tk.Tk()
    frecuencias_window.title("Frecuencias de Picos")

    # Crear un marco para mostrar las frecuencias y notas
    frame = tk.Frame(frecuencias_window)
    frame.pack()

    for i, peak in enumerate(peaks[:5]):
        frecuencia = freq[peak]
        nota = frecuencia_a_nota(frecuencia)
        texto = f"F{i + 1}: {frecuencia:.2f} Hz - Nota: {nota}"
        label = tk.Label(frame, text=texto, padx=10, pady=5)
        label.pack()
    
    
def frecuencia_a_nota(frecuencia):
    notas_piano = {
        "A0": (27.50, 29.13),
        "A#0/Bb0": (29.14, 30.86),
        "B0": (30.87, 32.69),
        "C1": (32.70, 34.64),
        "C#1/Db1": (34.65, 36.70),
        "D1": (36.71, 38.88),
        "D#1/Eb1": (38.89, 41.19),
        "E1": (41.20, 43.64),
        "F1": (43.65, 46.24),
        "F#1/Gb1": (46.25, 48.99),
        "G1": (49.00, 51.90),
        "G#1/Ab1": (51.91, 54.99),
        "A1": (55.00, 58.26),
        "A#1/Bb1": (58.27, 61.73),
        "B1": (61.74, 65.40),
        "C2": (65.41, 69.29),
        "C#2/Db2": (69.30, 73.41),
        "D2": (73.42, 77.77),
        "D#2/Eb2": (77.78, 82.40),
        "E2": (82.41, 87.30),
        "F2": (87.31, 92.49),
        "F#2/Gb2": (92.50, 97.99),
        "G2": (98.00, 103.82),
        "G#2/Ab2": (103.83, 110.00),
        "A2": (110.00, 116.53),
        "A#2/Bb2": (116.54, 123.46),
        "B2": (123.47, 130.80),
        "C3": (130.81, 138.58),
        "C#3/Db3": (138.59, 146.82),
        "D3": (146.83, 155.55),
        "D#3/Eb3": (155.56, 164.80),
        "E3": (164.81, 174.60),
        "F3": (174.61, 185.00),
        "F#3/Gb3": (185.00, 196.00),
        "G3": (196.00, 207.64),
        "G#3/Ab3": (207.65, 220.00),
        "A3": (220.00, 233.07),
        "A#3/Bb3": (233.08, 246.93),
        "B3": (246.94, 261.62),
        "C4 (Middle C)": (261.63, 277.17),
        "C#4/Db4": (277.18, 293.65),
        "D4": (293.66, 311.12),
        "D#4/Eb4": (311.13, 329.62),
        "E4": (329.63, 349.22),
        "F4": (349.23, 369.98),
        "F#4/Gb4": (369.99, 391.99),
        "G4": (392.00, 415.29),
        "G#4/Ab4": (415.30, 440.00),
        "A4": (440.00, 466.15),
        "A#4/Bb4": (466.16, 493.87),
        "B4": (493.88, 523.24),
        "C5": (523.25, 554.36),
        "C#5/Db5": (554.37, 587.32),
        "D5": (587.33, 622.24),
        "D#5/Eb5": (622.25, 659.25),
        "E5": (659.26, 698.45),
        "F5": (698.46, 739.98),
        "F#5/Gb5": (739.99, 783.98),
        "G5": (783.99, 830.60),
        "G#5/Ab5": (830.61, 880.00),
        "A5": (880.00, 932.32),
        "A#5/Bb5": (932.33, 987.76),
        "B5": (987.77, 1046.49),
        "C6": (1046.50, 1108.72),
        "C#6/Db6": (1108.73, 1174.65),
        "D6": (1174.66, 1244.50),
        "D#6/Eb6": (1244.51, 1318.50),
        "E6": (1318.51, 1396.90),
        "F6": (1396.91, 1479.97),
        "F#6/Gb6": (1479.98, 1567.97),
        "G6": (1567.98, 1661.21),
        "G#6/Ab6": (1661.22, 1760.00),
        "A6": (1760.00, 1864.65),
        "A#6/Bb6": (1864.66, 1975.52),
        "B6": (1975.53, 2093.00),
        "C7": (2093.00, 2217.45),
        "C#7/Db7": (2217.46, 2349.31),
        "D7": (2349.32, 2489.01),
        "D#7/Eb7": (2489.02, 2637.01),
        "E7": (2637.02, 2793.82),
        "F7": (2793.83, 2959.95),
        "F#7/Gb7": (2959.96, 3135.95),
        "G7": (3135.96, 3322.43),
        "G#7/Ab7": (3322.44, 3520.00),
        "A7": (3520.00, 3729.30),
        "A#7/Bb7": (3729.31, 3951.06),
        "B7": (3951.07, 3975.05),
        "C8": (3975.06, 4186.00)
    }

    for nota, (min_freq, max_freq) in notas_piano.items():
        if min_freq <= frecuencia <= max_freq:
            return nota

    # Si la frecuencia no coincide con ninguna nota en el piano, se devuelve  "nota no encontrada.
    return "Nota no encontrada"

#permite obtener las graficas al darle clic al boton graficas
def graficas():
    # Agrega aquí la lógica para la función "Gráficas"
    global x_senal1
    x_senal1 = x_senal1 / float(np.max(np.abs(x_senal1)))
    t = np.arange(0, float(len(x_senal1)) / fs, 1.0 / fs)    
    ventanas_graficas(t,x_senal1)
    

#Permite salir de la aplicacion
def salir():
    root.destroy()



# Crear la ventana principal
root = tk.Tk()
root.title("Mi Aplicación")
root.geometry("400x300")  # Ajusta el tamaño de la ventana (ancho x alto)

# Botones
btn_agregar = tk.Button(root, text="Añadir", command=agregar, width=20, height=2)
btn_agregar.pack(pady=10)  # Agrega un espacio vertical entre botones

btn_reproducir = tk.Button(root, text="Reproducir", command=reproducir, width=20, height=2 , state=tk.DISABLED)
btn_reproducir.pack(pady=10)


btn_graficas = tk.Button(root, text="Gráficas", command=graficas, width=20, height=2, state=tk.DISABLED)
btn_graficas.pack(pady=10)

btn_salir = tk.Button(root, text="Salir", command=salir, width=20, height=2, bg="red", fg="white")
btn_salir.pack(pady=10)

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from sympy import symbols, lambdify, sympify
from sympy.parsing.sympy_parser import parse_expr
from scipy.optimize import fsolve


# ========== CONFIGURACIÓN DE LA PÁGINA ==========
st.set_page_config(
    page_title="Métodos Numéricos",
    page_icon="",
    layout="wide"
)

# ========== FUNCIONES AUXILIARES ==========

# Diccionario de funciones seguras para eval
safe_funcs = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e
}

user_funcs = safe_funcs.copy()

def limpiar_input(texto):
    """Convierte ^ a ** para potencias"""
    return texto.replace("^", "**")

# ===================================
# FUNCIONES PARA SISTEMAS NO LINEALES
# ===================================

# FUNCION DEL METODO DE BISECCION
def metodo_biseccion_ui():
    st.subheader(" Método de Bisección")
    
    st.markdown("""
    **Descripción:** Encuentra la raíz de una ecuación en un intervalo [a,b] donde f(a) y f(b) tienen signos opuestos.
    
    **Fórmula:** $c = \\frac{a + b}{2}$
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ecuacion = st.text_input(
            "Ingrese la función en x:",
            value="x**3 - x - 2",
            help="Ejemplos: x**3 - x - 2, sin(x) - x/2, exp(x) - 3*x"
        )
    
    with col2:
        tol = st.number_input("Tolerancia:", value=0.00000001, format="%.8f")
    
    col_a, col_b = st.columns(2)
    with col_a:
        a = st.number_input("Límite inferior (a):", value=1.0)
    with col_b:
        b = st.number_input("Límite superior (b):", value=2.0)
    
    if st.button(" Calcular Bisección", type="primary"):
        try:
            x = symbols('x')
            f = sympify(ecuacion)
            f_lambda = lambdify(x, f, 'numpy')
            
            if f_lambda(a) * f_lambda(b) > 0:
                st.error(" No hay cambio de signo en el intervalo [a,b]. Intenta otros valores.")
                return
            
            iteraciones = []
            a_original, b_original = a, b
            iteracion = 0
            c = None
            
            while abs(b - a) > tol:
                iteracion += 1
                c = (a + b) / 2
                fc = f_lambda(c)
                
                iteraciones.append({
                    "Iteración": iteracion,
                    "a": round(a, 8),
                    "b": round(b, 8),
                    "c": round(c, 8),
                    "f(c)": round(fc, 8),
                    "Error": round(abs(b - a), 8)
                })
                
                if f_lambda(a) * fc < 0:
                    b = c
                else:
                    a = c
                
                if abs(fc) < tol:
                    break
            
            # Mostrar resultados
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(" Raíz aproximada", f"{c:.4f}")
            with col_m2:
                st.metric(" Iteraciones", iteracion)
            with col_m3:
                st.metric(" Tolerancia", f"{tol:.4f}")
            
            # Gráfica y tabla en columnas
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Gráfica de la función")
                X = np.linspace(a_original - 1, b_original + 1, 400)
                Y = f_lambda(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.axhline(0, color='black', lw=0.8, linestyle='--')
                ax.axvline(0, color='black', lw=0.8, linestyle='--')
                ax.plot(X, Y, 'b-', linewidth=2, label=f"f(x) = {ecuacion}")
                ax.scatter(c, f_lambda(c), color='red', s=150, zorder=5, label=f"Raíz ≈ {c:.4f}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=10)
                ax.set_xlabel("x", fontsize=12)
                ax.set_ylabel("f(x)", fontsize=12)
                ax.set_title("Método de Bisección", fontsize=14)
                st.pyplot(fig)
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(iteraciones)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="biseccion_resultados.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DEL METODO DE SECANTE
def metodo_secante_ui():
    st.subheader(" Método de la Secante")
    
    st.markdown("""
    **Descripción:** Similar a Newton-Raphson pero no requiere la derivada. Usa dos puntos iniciales.
    
    **Fórmula:** $x_{n+1} = x_n - f(x_n) \\cdot \\frac{x_n - x_{n-1}}{f(x_n) - f(x_{n-1})}$
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ecuacion = st.text_input(
            "Ingrese la función en x:",
            value="x**3 - x - 2",
            key="secante_ec"
        )
    
    with col2:
        tol = st.number_input("Tolerancia:", value=0.00000001, format="%.8f", key="secante_tol")
    
    col_x0, col_x1, col_max = st.columns(3)
    with col_x0:
        x0 = st.number_input("Primer valor inicial (x₀):", value=1.0)
    with col_x1:
        x1 = st.number_input("Segundo valor inicial (x₁):", value=2.0)
    with col_max:
        max_iter = st.number_input("Máximo de iteraciones:", value=100, min_value=1, step=1)
    
    if st.button(" Calcular Secante", type="primary"):
        try:
            x = symbols('x')
            f = sympify(ecuacion)
            f_lambda = lambdify(x, f, 'numpy')
            
            iteraciones = []
            iteracion = 0
            error = abs(x1 - x0)
            x2 = x1
            
            while error > tol and iteracion < max_iter:
                iteracion += 1
                f0 = f_lambda(x0)
                f1 = f_lambda(x1)
                
                if abs(f1 - f0) < 1e-12:
                    st.error(" División por cero detectada. El método falla.")
                    return
                
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
                error = abs(x2 - x1)
                
                iteraciones.append({
                    "Iteración": iteracion,
                    "x₀": round(x0, 8),
                    "x₁": round(x1, 8),
                    "f(x₁)": round(f1, 8),
                    "x₂": round(x2, 8),
                    "Error": round(error, 8)
                })
                
                x0, x1 = x1, x2
            
            # Resultados
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(" Raíz aproximada", f"{x2:.4f}")
            with col_m2:
                st.metric(" Iteraciones", iteracion)
            with col_m3:
                estado = " Convergió" if error < tol else "️ No convergió"
                st.metric("Estado", estado)
            
            # Gráfica y tabla
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Gráfica de la función")
                X = np.linspace(x2 - 3, x2 + 3, 400)
                Y = f_lambda(X)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.axhline(0, color='black', lw=0.8, linestyle='--')
                ax.plot(X, Y, 'b-', linewidth=2, label=f"f(x) = {ecuacion}")
                ax.scatter(x2, f_lambda(x2), color='red', s=150, zorder=5, label=f"Raíz ≈ {x2:.4f}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_xlabel("x")
                ax.set_ylabel("f(x)")
                ax.set_title("Método de la Secante")
                st.pyplot(fig)
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(iteraciones)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="secante_resultados.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DEL METODO NEWTON RAPHSON
def newton_raphson_2v_ui():
    st.subheader(" Newton-Raphson (2 variables)")
    
    st.markdown("""
    **Descripción:** Resuelve sistemas de 2 ecuaciones no lineales con 2 incógnitas.
    
    Ejemplo de sistema:
    - f₁(x,y) = x² + y² - 4
    - f₂(x,y) = x - y - 1
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        f1_str = st.text_input("f₁(x,y) =", value="x**2 + y**2 - 4", help="Primera ecuación")
    with col2:
        f2_str = st.text_input("f₂(x,y) =", value="x - y - 1", help="Segunda ecuación")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x0_init = st.number_input("Valor inicial x₀:", value=1.0)
    with col_y:
        y0_init = st.number_input("Valor inicial y₀:", value=1.0)
    
    if st.button(" Calcular Newton-Raphson 2V", type="primary"):
        try:
            x, y = sp.symbols("x y")
            
            f1_expr = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
            f2_expr = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})
            
            f1 = lambdify((x, y), f1_expr, "numpy")
            f2 = lambdify((x, y), f2_expr, "numpy")
            
            def sistema(vars):
                return [f1(vars[0], vars[1]), f2(vars[0], vars[1])]
            
            guess = [x0_init, y0_init]
            sol = fsolve(sistema, guess)
            
            st.success(f" Solución encontrada: x = {sol[0]:.4f}, y = {sol[1]:.4f}")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("x", f"{sol[0]:.4f}")
            with col_m2:
                st.metric("y", f"{sol[1]:.4f}")
            
            # Gráfica
            st.subheader(" Curvas de nivel f₁=0 y f₂=0")
            
            x_vals = np.linspace(sol[0] - 3, sol[0] + 3, 200)
            y_vals = np.linspace(sol[1] - 3, sol[1] + 3, 200)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z1 = f1(X, Y)
            Z2 = f2(X, Y)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='f₁=0')
            ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='f₂=0')
            ax.scatter(sol[0], sol[1], color='black', s=150, zorder=5, label='Solución')
            ax.scatter(x0_init, y0_init, color='green', s=100, marker='x', label='Punto inicial')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.set_title("Newton-Raphson (2 variables)")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DEL METODO NEWTON RAPHSON PARA 3 VARIABLES
def newton_raphson_3v_ui():
    st.subheader(" Newton-Raphson (3 variables)")
    
    st.markdown("""
    **Descripción:** Resuelve sistemas de 3 ecuaciones no lineales con 3 incógnitas.
    """)
    
    f1_str = st.text_input("f₁(x,y,z) =", value="x**2 + y**2 + z**2 - 9")
    f2_str = st.text_input("f₂(x,y,z) =", value="x + y - z - 1")
    f3_str = st.text_input("f₃(x,y,z) =", value="x - y + z - 1")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x0_init = st.number_input("x₀:", value=1.0)
    with col_y:
        y0_init = st.number_input("y₀:", value=1.0)
    with col_z:
        z0_init = st.number_input("z₀:", value=1.0)
    
    if st.button(" Calcular Newton-Raphson 3V", type="primary"):
        try:
            x, y, z = sp.symbols("x y z")
            
            f1_expr = parse_expr(f1_str, {"x":x, "y":y, "z":z, **user_funcs})
            f2_expr = parse_expr(f2_str, {"x":x, "y":y, "z":z, **user_funcs})
            f3_expr = parse_expr(f3_str, {"x":x, "y":y, "z":z, **user_funcs})
            
            f1 = lambdify((x,y,z), f1_expr, "numpy")
            f2 = lambdify((x,y,z), f2_expr, "numpy")
            f3 = lambdify((x,y,z), f3_expr, "numpy")
            
            def sistema(vars):
                return [f1(vars[0],vars[1],vars[2]), 
                        f2(vars[0],vars[1],vars[2]), 
                        f3(vars[0],vars[1],vars[2])]
            
            guess = [x0_init, y0_init, z0_init]
            sol = fsolve(sistema, guess)
            
            st.success(f" Solución: x={sol[0]:.4f}, y={sol[1]:.4f}, z={sol[2]:.4f}")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("x", f"{sol[0]:.4f}")
            with col_m2:
                st.metric("y", f"{sol[1]:.4f}")
            with col_m3:
                st.metric("z", f"{sol[2]:.4f}")
            
            # Gráfica 3D de trayectoria
            st.subheader(" Trayectoria de convergencia")
            
            from mpl_toolkits.mplot3d import Axes3D
            pts = [guess]
            v = np.array(guess)
            for _ in range(8):
                v = v + 0.5 * (np.array(sol) - v)
                pts.append(v.copy())
            pts = np.array(pts)
            
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(pts[:,0], pts[:,1], pts[:,2], marker='o', linewidth=2, markersize=6)
            ax.scatter(sol[0], sol[1], sol[2], color='red', s=150, label='Solución')
            ax.scatter(guess[0], guess[1], guess[2], color='green', s=100, marker='x', label='Inicial')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title('Newton-Raphson (3 variables)')
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DEL METODO NEWTON MODIFICADO PARA 2 VARIABLES
def newton_modificado_2v_ui():
    st.subheader(" Newton-Raphson Modificado (2 variables)")
    
    st.markdown("""
    **Descripción:** Versión modificada de Newton-Raphson que calcula explícitamente la matriz Jacobiana.
    
    Usa la fórmula: $X_{n+1} = X_n - J^{-1}(X_n) \\cdot F(X_n)$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        f1_str = st.text_input("f₁(x,y) =", value="x**2 + y**2 - 4", key="nm2v_f1")
    with col2:
        f2_str = st.text_input("f₂(x,y) =", value="x - y - 1", key="nm2v_f2")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x0_init = st.number_input("Valor inicial x₀:", value=1.0, key="nm2v_x0")
    with col_y:
        y0_init = st.number_input("Valor inicial y₀:", value=1.0, key="nm2v_y0")
    
    col_tol, col_max = st.columns(2)
    with col_tol:
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.2e", key="nm2v_tol")
    with col_max:
        max_iter = st.number_input("Máximo iteraciones:", value=100, min_value=1, step=1, key="nm2v_max")
    
    if st.button(" Calcular Newton Modificado 2V", type="primary"):
        try:
            x, y = sp.symbols("x y")
            
            # Limpiar entrada
            f1_str_clean = limpiar_input(f1_str)
            f2_str_clean = limpiar_input(f2_str)
            
            # Parsear expresiones
            f1 = parse_expr(f1_str_clean, {"x": x, "y": y, **user_funcs})
            f2 = parse_expr(f2_str_clean, {"x": x, "y": y, **user_funcs})
            
            # Crear vector F y calcular Jacobiano
            Fx = sp.Matrix([f1, f2])
            vars_vec = sp.Matrix([x, y])
            J = Fx.jacobian(vars_vec)
            
            # Lambdify para evaluación numérica
            f_lamb = sp.lambdify((x, y), Fx, "numpy")
            J_lamb = sp.lambdify((x, y), J, "numpy")
            
            # Inicializar
            x0, y0 = x0_init, y0_init
            datos = []
            
            # Iteraciones
            for i in range(int(max_iter)):
                Fv = np.array(f_lamb(x0, y0), dtype=float).reshape(2, 1)
                Jv = np.array(J_lamb(x0, y0), dtype=float)
                
                try:
                    delta = np.linalg.solve(Jv, -Fv)
                except np.linalg.LinAlgError:
                    st.error(" Error: matriz Jacobiana singular.")
                    return
                
                x1, y1 = (np.array([x0, y0]) + delta.flatten())
                error = np.linalg.norm(delta)
                
                datos.append({
                    "Iteración": i + 1,
                    "x₀": round(x0, 8),
                    "y₀": round(y0, 8),
                    "x₁": round(x1, 8),
                    "y₁": round(y1, 8),
                    "Error": round(error, 10)
                })
                
                if error < tol:
                    x0, y0 = x1, y1
                    break
                
                x0, y0 = x1, y1
            
            # Mostrar resultados
            if error < tol:
                st.success(f" Convergió en {i+1} iteraciones")
            else:
                st.warning(f"️ No convergió completamente en {max_iter} iteraciones")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("x", f"{x0:.4f}")
            with col_m2:
                st.metric("y", f"{y0:.4f}")
            with col_m3:
                st.metric("Iteraciones", i+1)
            
            # Gráfica y tabla
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Curvas de nivel")
                
                try:
                    x_vals = np.linspace(x0 - 3, x0 + 3, 200)
                    y_vals = np.linspace(y0 - 3, y0 + 3, 200)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    Z1 = np.vectorize(lambda x, y: eval(f1_str_clean, {"x": x, "y": y, **safe_funcs}))(X, Y)
                    Z2 = np.vectorize(lambda x, y: eval(f2_str_clean, {"x": x, "y": y, **safe_funcs}))(X, Y)
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='f₁=0')
                    ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='f₂=0')
                    ax.scatter(x0, y0, color='black', s=150, zorder=5, label='Solución')
                    ax.scatter(x0_init, y0_init, color='green', s=100, marker='x', label='Inicial')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.legend()
                    ax.set_title("Newton-Raphson Modificado (2V)")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f" No se pudo graficar: {str(e)}")
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(datos)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="newton_modificado_2v.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

#FUNCION DEL METODO NEWTON MODIFICADO PARA 3 VARIABLES

def newton_modificado_3v_ui():
    st.subheader(" Newton-Raphson Modificado (3 variables)")
    
    st.markdown("""
    **Descripción:** Versión modificada de Newton-Raphson para 3 variables con cálculo explícito del Jacobiano.
    """)
    
    f1_str = st.text_input("f₁(x,y,z) =", value="x**2 + y**2 + z**2 - 9", key="nm3v_f1")
    f2_str = st.text_input("f₂(x,y,z) =", value="x + y - z - 1", key="nm3v_f2")
    f3_str = st.text_input("f₃(x,y,z) =", value="x - y + z - 1", key="nm3v_f3")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x0_init = st.number_input("x₀:", value=1.0, key="nm3v_x0")
    with col_y:
        y0_init = st.number_input("y₀:", value=1.0, key="nm3v_y0")
    with col_z:
        z0_init = st.number_input("z₀:", value=1.0, key="nm3v_z0")
    
    col_tol, col_max = st.columns(2)
    with col_tol:
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.2e", key="nm3v_tol")
    with col_max:
        max_iter = st.number_input("Máximo iteraciones:", value=100, min_value=1, step=1, key="nm3v_max")
    
    if st.button(" Calcular Newton Modificado 3V", type="primary"):
        try:
            x, y, z = sp.symbols("x y z")
            
            # Limpiar entrada
            f1_str_clean = limpiar_input(f1_str)
            f2_str_clean = limpiar_input(f2_str)
            f3_str_clean = limpiar_input(f3_str)
            
            # Parsear expresiones
            f1 = parse_expr(f1_str_clean, {"x": x, "y": y, "z": z, **user_funcs})
            f2 = parse_expr(f2_str_clean, {"x": x, "y": y, "z": z, **user_funcs})
            f3 = parse_expr(f3_str_clean, {"x": x, "y": y, "z": z, **user_funcs})
            
            # Crear vector F y calcular Jacobiano
            Fx = sp.Matrix([f1, f2, f3])
            vars_vec = sp.Matrix([x, y, z])
            J = Fx.jacobian(vars_vec)
            
            # Lambdify para evaluación numérica
            f_lamb = sp.lambdify((x, y, z), Fx, "numpy")
            J_lamb = sp.lambdify((x, y, z), J, "numpy")
            
            # Inicializar
            x0, y0, z0 = x0_init, y0_init, z0_init
            datos = []
            pts = [[x0, y0, z0]]
            
            # Iteraciones
            for i in range(int(max_iter)):
                Fv = np.array(f_lamb(x0, y0, z0), dtype=float).reshape(3, 1)
                Jv = np.array(J_lamb(x0, y0, z0), dtype=float)
                
                try:
                    delta = np.linalg.solve(Jv, -Fv)
                except np.linalg.LinAlgError:
                    st.error(" Error: matriz Jacobiana singular.")
                    return
                
                x1, y1, z1 = (np.array([x0, y0, z0]) + delta.flatten())
                error = np.linalg.norm(delta)
                
                datos.append({
                    "Iteración": i + 1,
                    "x₀": round(x0, 8),
                    "y₀": round(y0, 8),
                    "z₀": round(z0, 8),
                    "x₁": round(x1, 8),
                    "y₁": round(y1, 8),
                    "z₁": round(z1, 8),
                    "Error": round(error, 10)
                })
                
                pts.append([x1, y1, z1])
                
                if error < tol:
                    x0, y0, z0 = x1, y1, z1
                    break
                
                x0, y0, z0 = x1, y1, z1
            
            # Mostrar resultados
            if error < tol:
                st.success(f" Convergió en {i+1} iteraciones")
            else:
                st.warning(f"️ No convergió completamente en {max_iter} iteraciones")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("x", f"{x0:.4f}")
            with col_m2:
                st.metric("y", f"{y0:.4f}")
            with col_m3:
                st.metric("z", f"{z0:.4f}")
            with col_m4:
                st.metric("Iteraciones", i+1)
            
            # Gráfica y tabla
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Trayectoria 3D")
                
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    pts = np.array(pts)
                    
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', color='blue', linewidth=2, markersize=6, label="Trayectoria")
                    ax.scatter(x0, y0, z0, color='red', s=150, label='Solución')
                    ax.scatter(x0_init, y0_init, z0_init, color='green', s=100, marker='x', label='Inicial')
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    ax.set_title("Newton-Raphson Modificado (3V)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f" No se pudo graficar: {str(e)}")
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(datos)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="newton_modificado_3v.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DE PUNTO FIJO PARA 2 VARIABLES

def punto_fijo_2v_ui():
    st.subheader(" Punto Fijo (2 variables)")
    
    st.markdown("""
    **Descripción:** Método iterativo donde x = g₁(x,y) y y = g₂(x,y).
    
    **Fórmula:** $X_{n+1} = G(X_n)$ donde $G = [g_1(x,y), g_2(x,y)]^T$
    """)
    
    st.info("Ingresa las funciones de iteración (no las ecuaciones originales)")
    
    col1, col2 = st.columns(2)
    with col1:
        fx = st.text_input("x = g₁(x,y)", value="(4 - y**2) / 2", help="Función de iteración para x", key="pf2_fx")
    with col2:
        fy = st.text_input("y = g₂(x,y)", value="(x + 1)", help="Función de iteración para y", key="pf2_fy")
    
    col_x, col_y = st.columns(2)
    with col_x:
        x0 = st.number_input("Valor inicial x₀:", value=1.0, key="pf2_x0")
    with col_y:
        y0 = st.number_input("Valor inicial y₀:", value=1.0, key="pf2_y0")
    
    col_tol, col_max = st.columns(2)
    with col_tol:
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.2e", key="pf2_tol")
    with col_max:
        max_iter = st.number_input("Máximo iteraciones:", value=100, min_value=1, step=1, key="pf2_max")
    
    if st.button(" Calcular Punto Fijo 2V", type="primary"):
        try:
            # Limpiar entrada
            fx_clean = limpiar_input(fx)
            fy_clean = limpiar_input(fy)
            
            # Función de iteración
            def g(v):
                x, y = v
                locals_dict = {"x": x, "y": y, **safe_funcs}
                return np.array([
                    eval(fx_clean, {"__builtins__": None}, locals_dict),
                    eval(fy_clean, {"__builtins__": None}, locals_dict)
                ])
            
            # Iteraciones
            datos = []
            v = np.array([x0, y0])
            convergio = False
            
            for i in range(int(max_iter)):
                try:
                    nuevo = g(v)
                    error = np.linalg.norm(nuevo - v)
                    
                    datos.append({
                        "Iteración": i + 1,
                        "x₀": round(v[0], 8),
                        "y₀": round(v[1], 8),
                        "x₁": round(nuevo[0], 8),
                        "y₁": round(nuevo[1], 8),
                        "Error": round(error, 10)
                    })
                    
                    if error < tol:
                        v = nuevo
                        convergio = True
                        break
                    
                    v = nuevo
                    
                except Exception as e:
                    st.error(f" Error en iteración {i+1}: {str(e)}")
                    break
            
            # Mostrar resultados
            if convergio:
                st.success(f" Convergió en {i+1} iteraciones")
            else:
                st.warning(f"️ No alcanzó la tolerancia en {max_iter} iteraciones")
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("x", f"{v[0]:.4f}")
            with col_m2:
                st.metric("y", f"{v[1]:.4f}")
            with col_m3:
                st.metric("Iteraciones", len(datos))
            
            # Gráfica y tabla
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Curvas x = g₁(x,y) y y = g₂(x,y)")
                
                try:
                    x_vals = np.linspace(v[0] - 3, v[0] + 3, 200)
                    y_vals = np.linspace(v[1] - 3, v[1] + 3, 200)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    
                    # Calcular g₁(x,y) - x y g₂(x,y) - y para graficar donde son cero
                    Z1 = np.vectorize(lambda x, y: eval(fx_clean, {"__builtins__": None, "x": x, "y": y, **safe_funcs}))(X, Y) - X
                    Z2 = np.vectorize(lambda x, y: eval(fy_clean, {"__builtins__": None, "x": x, "y": y, **safe_funcs}))(X, Y) - Y
                    
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2, label='x = g₁(x,y)')
                    ax.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2, label='y = g₂(x,y)')
                    ax.scatter(v[0], v[1], color='black', s=150, zorder=5, label='Punto fijo')
                    ax.scatter(x0, y0, color='green', s=100, marker='x', label='Inicial')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.legend()
                    ax.set_title("Método de Punto Fijo (2V)")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f" No se pudo graficar: {str(e)}")
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(datos)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="punto_fijo_2v.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

# FUNCION DE PUNTO FIJO PARA 3 VARIABLES

def punto_fijo_3v_ui():
    st.subheader(" Punto Fijo (3 variables)")
    
    st.markdown("""
    **Descripción:** Método iterativo para 3 variables donde x = g₁(x,y,z), y = g₂(x,y,z), z = g₃(x,y,z).
    """)
    
    st.info("Ingresa las funciones de iteración")
    
    fx = st.text_input("x = g₁(x,y,z)", value="(9 - y**2 - z**2) / 3", key="pf3_fx")
    fy = st.text_input("y = g₂(x,y,z)", value="(1 + x - z) / 2", key="pf3_fy")
    fz = st.text_input("z = g₃(x,y,z)", value="(1 + x - y) / 2", key="pf3_fz")
    
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x0 = st.number_input("x₀:", value=1.0, key="pf3_x0")
    with col_y:
        y0 = st.number_input("y₀:", value=1.0, key="pf3_y0")
    with col_z:
        z0 = st.number_input("z₀:", value=1.0, key="pf3_z0")
    
    col_tol, col_max = st.columns(2)
    with col_tol:
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.2e", key="pf3_tol")
    with col_max:
        max_iter = st.number_input("Máximo iteraciones:", value=100, min_value=1, step=1, key="pf3_max")
    
    if st.button(" Calcular Punto Fijo 3V", type="primary"):
        try:
            # Limpiar entrada
            fx_clean = limpiar_input(fx)
            fy_clean = limpiar_input(fy)
            fz_clean = limpiar_input(fz)
            
            # Función de iteración
            def g(v):
                x, y, z = v
                locals_dict = {"x": x, "y": y, "z": z, **safe_funcs}
                return np.array([
                    eval(fx_clean, {"__builtins__": None}, locals_dict),
                    eval(fy_clean, {"__builtins__": None}, locals_dict),
                    eval(fz_clean, {"__builtins__": None}, locals_dict)
                ])
            
            # Iteraciones
            datos = []
            v = np.array([x0, y0, z0])
            pts = [v.copy()]
            convergio = False
            
            for i in range(int(max_iter)):
                try:
                    nuevo = g(v)
                    error = np.linalg.norm(nuevo - v)
                    
                    datos.append({
                        "Iteración": i + 1,
                        "x₀": round(v[0], 8),
                        "y₀": round(v[1], 8),
                        "z₀": round(v[2], 8),
                        "x₁": round(nuevo[0], 8),
                        "y₁": round(nuevo[1], 8),
                        "z₁": round(nuevo[2], 8),
                        "Error": round(error, 10)
                    })
                    
                    v = nuevo
                    pts.append(v.copy())
                    
                    if error < tol:
                        convergio = True
                        break
                    
                except Exception as e:
                    st.error(f" Error en iteración {i+1}: {str(e)}")
                    break
            
            # Mostrar resultados
            if convergio:
                st.success(f" Convergió en {i+1} iteraciones")
            else:
                st.warning(f"️ No alcanzó la tolerancia en {max_iter} iteraciones")
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("x", f"{v[0]:.4f}")
            with col_m2:
                st.metric("y", f"{v[1]:.4f}")
            with col_m3:
                st.metric("z", f"{v[2]:.4f}")
            with col_m4:
                st.metric("Iteraciones", len(datos))
            
            # Gráfica y tabla
            col_graf, col_tab = st.columns([1.5, 1])
            
            with col_graf:
                st.subheader(" Trayectoria de convergencia 3D")
                
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    pts = np.array(pts)
                    
                    fig = plt.figure(figsize=(10, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', color='blue', linewidth=2, markersize=6, label="Trayectoria")
                    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='red', s=150, label='Punto fijo')
                    ax.scatter(x0, y0, z0, color='green', s=100, marker='x', label='Inicial')
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                    ax.set_title("Método de Punto Fijo (3V)")
                    ax.legend()
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f" No se pudo graficar: {str(e)}")
            
            with col_tab:
                st.subheader(" Tabla de iteraciones")
                df = pd.DataFrame(datos)
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="️ Descargar CSV",
                    data=csv,
                    file_name="punto_fijo_3v.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f" Error: {str(e)}")

# ======================
# SISTEMAS DE ECUACIONES
# ======================

def sistemas_lineales_ui():
    st.subheader(" Sistemas de Ecuaciones Lineales")
    
    st.markdown("""
    **Resuelve sistemas de la forma Ax = b**
    
    Puedes elegir entre métodos directos e iterativos.
    """)
    
    metodo = st.selectbox(
        "Selecciona el método:",
        ["Método de la Inversa", "Eliminación de Gauss", "Gauss-Jordan", 
         "Jacobi (iterativo)", "Gauss-Seidel (iterativo)"]
    )
    
    st.info(" Ingresa la matriz A y el vector b del sistema Ax = b")
    
    n = st.number_input("Tamaño del sistema (n×n):", min_value=2, max_value=10, value=3, step=1)
    
    st.write("**Matriz A:**")
    cols_A = st.columns(int(n))
    A = []
    for i in range(int(n)):
        fila = []
        for j in range(int(n)):
            with cols_A[j]:
                val = st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i==j else 0.0, key=f"a_{i}_{j}", format="%.4f")
                fila.append(val)
        A.append(fila)
    
    A = np.array(A)
    
    st.write("**Vector b:**")
    cols_b = st.columns(int(n))
    b = []
    for i in range(int(n)):
        with cols_b[i]:
            val = st.number_input(f"b[{i+1}]", value=1.0, key=f"b_{i}", format="%.4f")
            b.append(val)
    
    b = np.array(b)
    
    # Parámetros para métodos iterativos
    if metodo in ["Jacobi (iterativo)", "Gauss-Seidel (iterativo)"]:
        col_tol, col_max = st.columns(2)
        with col_tol:
            tol = st.number_input("Tolerancia:", value=1e-6, format="%.2e")
        with col_max:
            max_iter = st.number_input("Máximo iteraciones:", value=100, min_value=1, step=1)
    
    if st.button(" Resolver Sistema", type="primary"):
        try:
            if metodo == "Método de la Inversa":
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    st.error(" El sistema no tiene solución única (det(A) ≈ 0)")
                    return
                
                A_inv = np.linalg.inv(A)
                x = np.dot(A_inv, b)
                
                st.success(" Sistema resuelto usando A⁻¹")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Inversa de A:**")
                    st.dataframe(pd.DataFrame(A_inv), use_container_width=True)
                
                with col2:
                    st.write("**Solución x:**")
                    sol_df = pd.DataFrame({"Variable": [f"x{i+1}" for i in range(len(x))],
                                          "Valor": x})
                    st.dataframe(sol_df, use_container_width=True)
            
            elif metodo == "Eliminación de Gauss":
                n_size = len(b)
                M = np.hstack([A.astype(float), b.reshape(-1,1)])
                
                for k in range(n_size):
                    max_row = np.argmax(abs(M[k:,k])) + k
                    M[[k, max_row]] = M[[max_row, k]]
                    for i in range(k+1, n_size):
                        if abs(M[k][k]) < 1e-10:
                            st.error(" División por cero en eliminación")
                            return
                        factor = M[i][k] / M[k][k]
                        M[i] = M[i] - factor * M[k]
                
                x = np.zeros(n_size)
                for i in range(n_size-1, -1, -1):
                    x[i] = (M[i, -1] - np.dot(M[i,i+1:n_size], x[i+1:n_size])) / M[i,i]
                
                st.success(" Sistema resuelto usando Eliminación de Gauss")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Matriz aumentada final:**")
                    st.dataframe(pd.DataFrame(M), use_container_width=True)
                
                with col2:
                    st.write("**Solución x:**")
                    sol_df = pd.DataFrame({"Variable": [f"x{i+1}" for i in range(len(x))],
                                          "Valor": x})
                    st.dataframe(sol_df, use_container_width=True)
            
            elif metodo == "Gauss-Jordan":
                n_size = len(b)
                M = np.hstack([A.astype(float), b.reshape(-1,1)])
                
                for k in range(n_size):
                    if abs(M[k][k]) < 1e-10:
                        st.error(" Pivote cero en Gauss-Jordan")
                        return
                    M[k] = M[k] / M[k][k]
                    for i in range(n_size):
                        if i != k:
                            M[i] = M[i] - M[i][k] * M[k]
                
                x = M[:, -1]
                
                st.success(" Sistema resuelto usando Gauss-Jordan")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Matriz en forma identidad:**")
                    st.dataframe(pd.DataFrame(M), use_container_width=True)
                
                with col2:
                    st.write("**Solución x:**")
                    sol_df = pd.DataFrame({"Variable": [f"x{i+1}" for i in range(len(x))],
                                          "Valor": x})
                    st.dataframe(sol_df, use_container_width=True)
            
            elif metodo == "Jacobi (iterativo)":
                n_size = len(b)
                x = np.zeros(n_size)
                historial = []
                
                for it in range(int(max_iter)):
                    x_new = np.zeros(n_size)
                    for i in range(n_size):
                        if abs(A[i][i]) < 1e-10:
                            st.error(" Diagonal con cero en Jacobi")
                            return
                        s = sum(A[i][j] * x[j] for j in range(n_size) if j != i)
                        x_new[i] = (b[i] - s) / A[i][i]
                    
                    historial.append([it+1] + list(x_new))
                    
                    if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                        st.success(f" Convergió en {it+1} iteraciones")
                        df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n_size)])
                        
                        col_res, col_tab = st.columns([1, 2])
                        with col_res:
                            st.write("**Solución final:**")
                            sol_df = pd.DataFrame({"Variable": [f"x{i+1}" for i in range(n_size)],
                                                  "Valor": x_new})
                            st.dataframe(sol_df, use_container_width=True)
                        
                        with col_tab:
                            st.write("**Historial de iteraciones:**")
                            df = df.round(3)
                            st.dataframe(df, use_container_width=True, height=400)
                        return
                    
                    x = x_new
                
                st.warning(f"️ No convergió en {max_iter} iteraciones")
                df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n_size)])
                df = df.round(3)
                st.dataframe(df, use_container_width=True, height=400)
                
            elif metodo == "Gauss-Seidel (iterativo)":
                n_size = len(b)
                x = np.zeros(n_size)
                historial = []
                
                for it in range(int(max_iter)):
                    x_new = np.copy(x)
                    for i in range(n_size):
                        if abs(A[i][i]) < 1e-10:
                            st.error(" Diagonal con cero en Gauss-Seidel")
                            return
                        s = sum(A[i][j] * x_new[j] for j in range(n_size) if j != i)
                        x_new[i] = (b[i] - s) / A[i][i]
                    
                    historial.append([it+1] + list(x_new))
                    
                    if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                        st.success(f" Convergió en {it+1} iteraciones")
                        df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n_size)])
                        
                        col_res, col_tab = st.columns([1, 2])
                        with col_res:
                            st.write("**Solución final:**")
                            sol_df = pd.DataFrame({"Variable": [f"x{i+1}" for i in range(n_size)],
                                                "Valor": x_new})
                            st.dataframe(sol_df, use_container_width=True)
                        
                        with col_tab:
                            st.write("**Historial de iteraciones:**")
                            st.dataframe(df, use_container_width=True, height=400)
                        return
                    
                    x = x_new
                
                st.warning(f" No convergió en {max_iter} iteraciones")
                df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n_size)])
                st.dataframe(df, use_container_width=True, height=400)

        except Exception as e:
            st.error(f" Error al resolver el sistema: {str(e)}")


# =================
# ALGEBRA MATRICIAL
# =================

def algebra_matricial_ui():
    st.subheader(" Operaciones con Matrices")
    
    operacion = st.selectbox(
        "Selecciona la operación:",
        ["Suma de matrices", "Multiplicación de matrices", "Determinante", "Inversa de matriz"]
    )
    # SUMA DE MATRICES
    if operacion == "Suma de matrices":
        st.info(" Suma dos matrices del mismo tamaño")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Matriz A:**")
            filas_a = st.number_input("Filas de A:", min_value=1, max_value=10, value=2, step=1, key="filas_a")
            cols_a = st.number_input("Columnas de A:", min_value=1, max_value=10, value=2, step=1, key="cols_a")
            
            A = []
            for i in range(int(filas_a)):
                fila = []
                cols_input = st.columns(int(cols_a))
                for j in range(int(cols_a)):
                    with cols_input[j]:
                        val = st.number_input(f"A[{i+1},{j+1}]", value=0.0, key=f"suma_a_{i}_{j}", format="%.2f")
                        fila.append(val)
                A.append(fila)
            A = np.array(A)
        
        with col2:
            st.write("**Matriz B:**")
            st.write(f"Tamaño: {int(filas_a)}×{int(cols_a)}")
            
            B = []
            for i in range(int(filas_a)):
                fila = []
                cols_input = st.columns(int(cols_a))
                for j in range(int(cols_a)):
                    with cols_input[j]:
                        val = st.number_input(f"B[{i+1},{j+1}]", value=0.0, key=f"suma_b_{i}_{j}", format="%.2f")
                        fila.append(val)
                B.append(fila)
            B = np.array(B)
        
        if st.button(" Calcular Suma", type="primary"):
            resultado = A + B
            st.success(" Suma calculada")
            st.write("**Resultado: A + B =**")
            st.dataframe(pd.DataFrame(resultado), use_container_width=True)

    # MULTIPLICACION DE MATRICES

    elif operacion == "Multiplicación de matrices":
        st.info("️ Multiplica dos matrices (columnas de A = filas de B)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Matriz A:**")
            filas_a = st.number_input("Filas de A:", min_value=1, max_value=10, value=2, step=1, key="mult_filas_a")
            cols_a = st.number_input("Columnas de A:", min_value=1, max_value=10, value=3, step=1, key="mult_cols_a")
            
            A = []
            for i in range(int(filas_a)):
                fila = []
                cols_input = st.columns(int(cols_a))
                for j in range(int(cols_a)):
                    with cols_input[j]:
                        val = st.number_input(f"A[{i+1},{j+1}]", value=1.0, key=f"mult_a_{i}_{j}", format="%.2f")
                        fila.append(val)
                A.append(fila)
            A = np.array(A)
        
        with col2:
            st.write("**Matriz B:**")
            filas_b = int(cols_a)  # Debe coincidir con columnas de A
            st.write(f"Filas de B: {filas_b} (fijo)")
            cols_b = st.number_input("Columnas de B:", min_value=1, max_value=10, value=2, step=1, key="mult_cols_b")
            
            B = []
            for i in range(filas_b):
                fila = []
                cols_input = st.columns(int(cols_b))
                for j in range(int(cols_b)):
                    with cols_input[j]:
                        val = st.number_input(f"B[{i+1},{j+1}]", value=1.0, key=f"mult_b_{i}_{j}", format="%.2f")
                        fila.append(val)
                B.append(fila)
            B = np.array(B)
        
        if st.button("️ Calcular Multiplicación", type="primary"):
            resultado = np.dot(A, B)
            st.success(f" Multiplicación calculada (resultado: {resultado.shape[0]}×{resultado.shape[1]})")
            st.write("**Resultado: A × B =**")
            st.dataframe(pd.DataFrame(resultado), use_container_width=True)
    
    elif operacion == "Determinante":
        st.info(" Calcula el determinante de una matriz cuadrada")
        
        n = st.number_input("Tamaño de la matriz (n×n):", min_value=2, max_value=10, value=3, step=1, key="det_n")
        
        st.write("**Matriz A:**")
        A = []
        for i in range(int(n)):
            fila = []
            cols_input = st.columns(int(n))
            for j in range(int(n)):
                with cols_input[j]:
                    val = st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i==j else 0.0, key=f"det_a_{i}_{j}", format="%.2f")
                    fila.append(val)
            A.append(fila)
        A = np.array(A)
        
        if st.button(" Calcular Determinante", type="primary"):
            det = np.linalg.det(A)
            st.success(" Determinante calculado")
            st.metric("Determinante de A", f"{det:.6f}")
            
            if abs(det) < 1e-10:
                st.warning("️ La matriz es singular (no invertible)")
            else:
                st.info(" La matriz es invertible")

    # MATRIZ INVERSA
    
    elif operacion == "Inversa de matriz":
        st.info(" Calcula la matriz inversa de una matriz cuadrada")
        
        n = st.number_input("Tamaño de la matriz (n×n):", min_value=2, max_value=10, value=3, step=1, key="inv_n")
        
        st.write("**Matriz A:**")
        A = []
        for i in range(int(n)):
            fila = []
            cols_input = st.columns(int(n))
            for j in range(int(n)):
                with cols_input[j]:
                    val = st.number_input(f"A[{i+1},{j+1}]", value=1.0 if i==j else 0.0, key=f"inv_a_{i}_{j}", format="%.2f")
                    fila.append(val)
            A.append(fila)
        A = np.array(A)
        
        if st.button(" Calcular Inversa", type="primary"):
            try:
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    st.error("️ La matriz no es invertible (determinante ≈ 0)")
                else:
                    inv = np.linalg.inv(A)
                    st.success(" Inversa calculada")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Matriz original A:**")
                        st.dataframe(pd.DataFrame(A), use_container_width=True)
                    
                    with col2:
                        st.write("**Inversa A⁻¹:**")
                        st.dataframe(pd.DataFrame(inv), use_container_width=True)
                    
                    # Verificación
                    producto = np.dot(A, inv)
                    st.write("**Verificación: A × A⁻¹ ≈ I**")
                    st.dataframe(pd.DataFrame(producto), use_container_width=True)
            
            except np.linalg.LinAlgError:
                st.error(" Error: La matriz no es invertible")

def seleccionar_metodo(categoria, metodo=None):
    """Guarda la categoría y el método seleccionados en el estado de la sesión."""
    st.session_state.categoria = categoria
    if metodo:
        st.session_state.metodo_nl = metodo
    st.experimental_rerun()

# ===================================
# 4. INTERPOLACIÓN (NUEVO)
# ===================================

def polinomio_lagrange_ui():
    if "lagrange_calculado" not in st.session_state:
        st.session_state.lagrange_calculado = False
        st.session_state.P = None
        st.session_state.puntos = None

    st.markdown("<h3 style='text-align:center;'>Polinomio de Lagrange</h3>", unsafe_allow_html=True)
    n = st.number_input("Número de puntos:", min_value=2, max_value=20, value=3)
    puntos = []
    for i in range(n):
        c1, c2 = st.columns(2)
        with c1: xi = st.number_input(f"x{i}", key=f"xl_{i}")
        with c2: yi = st.number_input(f"y{i}", key=f"yl_{i}")
        puntos.append((xi, yi))

    if st.button("Calcular Lagrange"):
        x = sp.Symbol('x')
        P = 0
        for i in range(n):
            xi, yi = puntos[i]
            Li = 1
            for j in range(n):
                if i != j:
                    xj, _ = puntos[j]
                    Li *= (x - xj) / (xi - xj)
            P += yi * Li
        st.session_state.P = sp.expand(P)
        st.session_state.puntos = puntos
        st.session_state.lagrange_calculado = True

    if st.session_state.lagrange_calculado:
        P = st.session_state.P
        st.latex(sp.latex(P))
        xs_vals = [p[0] for p in st.session_state.puntos]
        ys_vals = [p[1] for p in st.session_state.puntos]
        
        # Gráfica
        X = np.linspace(min(xs_vals)-1, max(xs_vals)+1, 200)
        f = sp.lambdify(sp.Symbol('x'), P, "numpy")
        fig, ax = plt.subplots()
        ax.plot(X, f(X), label="Interpolación")
        ax.scatter(xs_vals, ys_vals, c='r', label="Datos")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

def newton_diferencias_divididas_ui():
    st.subheader("Newton - Diferencias Divididas")
    n = st.number_input("Cant. Puntos:", 2, 10, 3, key="ndd_n")
    x_vals, y_vals = [], []
    for i in range(n):
        c1, c2 = st.columns(2)
        with c1: x_vals.append(st.number_input(f"x{i}", key=f"ndd_x{i}"))
        with c2: y_vals.append(st.number_input(f"y{i}", key=f"ndd_y{i}"))
    
    if st.button("Interpolación Newton"):
        tabla = np.zeros((n, n))
        tabla[:, 0] = y_vals
        for j in range(1, n):
            for i in range(n - j):
                tabla[i][j] = (tabla[i+1][j-1] - tabla[i][j-1]) / (x_vals[i+j] - x_vals[i])
        
        st.write("Tabla de Diferencias:")
        st.dataframe(pd.DataFrame(tabla))
        
        x = sp.Symbol("x")
        P = tabla[0][0]
        for k in range(1, n):
            term = tabla[0][k]
            for j in range(k): term *= (x - x_vals[j])
            P += term
        st.latex(sp.latex(sp.expand(P)))

def minimos_cuadrados_ui():
    st.subheader("Ajuste por Mínimos Cuadrados")
    tipo = st.selectbox("Grado:", ["Lineal (1)", "Cuadrático (2)"])
    n = st.number_input("N datos:", 2, 50, 4)
    x_vals, y_vals = [], []
    for i in range(n):
        c1, c2 = st.columns(2)
        with c1: x_vals.append(st.number_input(f"x{i}", key=f"mc_x{i}"))
        with c2: y_vals.append(st.number_input(f"y{i}", key=f"mc_y{i}"))

    if st.button("Ajustar Curva"):
        xv, yv = np.array(x_vals), np.array(y_vals)
        grado = 1 if "Lineal" in tipo else 2
        coefs = np.polyfit(xv, yv, grado)
        polinomio = np.poly1d(coefs)
        
        st.write(f"Ecuación: {polinomio}")
        
        # Gráfica
        X = np.linspace(min(xv)-1, max(xv)+1, 200)
        fig, ax = plt.subplots()
        ax.scatter(xv, yv, color='red', label="Datos")
        ax.plot(X, polinomio(X), color='blue', label=f"Ajuste G{grado}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ===================================
# 5. INTEGRACIÓN (NUEVO)
# ===================================

def integracion_trapecio_ui():
    st.subheader("Método del Trapecio")
    func_str = st.text_input("f(x) =", "sin(x)", key="trap_f")
    c1, c2, c3 = st.columns(3)
    with c1: a = st.number_input("a:", value=0.0, key="trap_a")
    with c2: b = st.number_input("b:", value=np.pi, key="trap_b")
    with c3: n = st.number_input("Subintervalos (n):", 1, 100, 4, key="trap_n")
    
    if st.button("Integrar (Trapecio)"):
        try:
            x = sp.Symbol('x')
            f = lambdify(x, parse_expr(func_str), 'numpy')
            xs = np.linspace(a, b, n+1)
            ys = f(xs)
            h = (b - a) / n
            integral = (h/2) * (ys[0] + 2*np.sum(ys[1:-1]) + ys[-1])
            st.success(f"Integral ≈ {integral:.8f}")
            
            # Gráfica
            fig, ax = plt.subplots()
            x_fine = np.linspace(a, b, 200)
            ax.plot(x_fine, f(x_fine), 'b')
            for i in range(n):
                ax.fill([xs[i], xs[i], xs[i+1], xs[i+1]], [0, ys[i], ys[i+1], 0], 'b', alpha=0.2, edgecolor='k')
            st.pyplot(fig)
        except Exception as e: st.error(str(e))

def simpson_13_ui():
    st.subheader("Método de Simpson 1/3")
    st.info("Requiere n par.")
    func_str = st.text_input("f(x) =", "sin(x)", key="s13_f")
    c1, c2, c3 = st.columns(3)
    with c1: a = st.number_input("a:", value=0.0, key="s13_a")
    with c2: b = st.number_input("b:", value=np.pi, key="s13_b")
    with c3: n = st.number_input("n (par):", 2, 100, 4, step=2, key="s13_n")
    
    if st.button("Integrar (Simpson 1/3)"):
        try:
            if n % 2 != 0: st.error("n debe ser par"); return
            x = sp.Symbol('x')
            f = lambdify(x, parse_expr(func_str), 'numpy')
            h = (b - a) / n
            xs = np.linspace(a, b, n+1)
            ys = f(xs)
            integral = (h/3) * (ys[0] + 4*np.sum(ys[1:-1:2]) + 2*np.sum(ys[2:-1:2]) + ys[-1])
            st.success(f"Integral ≈ {integral:.8f}")
        except Exception as e: st.error(str(e))

def simpson_38_ui():
    st.subheader("Método de Simpson 3/8")
    st.info("Requiere n múltiplo de 3.")
    func_str = st.text_input("f(x) =", "sin(x)", key="s38_f")
    c1, c2, c3 = st.columns(3)
    with c1: a = st.number_input("a:", value=0.0, key="s38_a")
    with c2: b = st.number_input("b:", value=np.pi, key="s38_b")
    with c3: n = st.number_input("n (mult 3):", 3, 99, 3, step=3, key="s38_n")
    
    if st.button("Integrar (Simpson 3/8)"):
        try:
            if n % 3 != 0: st.error("n debe ser múltiplo de 3"); return
            x = sp.Symbol('x')
            f = lambdify(x, parse_expr(func_str), 'numpy')
            h = (b - a) / n
            xs = np.linspace(a, b, n+1)
            ys = f(xs)
            # Lógica suma Simpson 3/8
            suma = ys[0] + ys[-1]
            for i in range(1, n):
                if i % 3 == 0: suma += 2 * ys[i]
                else: suma += 3 * ys[i]
            integral = (3 * h / 8) * suma
            st.success(f"Integral ≈ {integral:.8f}")
        except Exception as e: st.error(str(e))

# ===================================
# 6. ECUACIONES DIFERENCIALES (NUEVO)
# ===================================

def euler_ui():
    st.subheader("Método de Euler")
    f_str = st.text_input("dy/dx = f(x,y)", "x + y", key="euler_f")
    c1, c2, c3, c4 = st.columns(4)
    with c1: x0 = st.number_input("x0", 0.0, key="eu_x0")
    with c2: y0 = st.number_input("y0", 1.0, key="eu_y0")
    with c3: h = st.number_input("paso h", 0.1, key="eu_h")
    with c4: xn = st.number_input("x final", 1.0, key="eu_xn")
    
    if st.button("Calcular Euler"):
        try:
            x, y = sp.symbols('x y')
            f = lambdify((x,y), parse_expr(f_str), 'numpy')
            xs, ys = [x0], [y0]
            while xs[-1] < xn - 1e-9:
                curr_x, curr_y = xs[-1], ys[-1]
                ys.append(curr_y + h * f(curr_x, curr_y))
                xs.append(curr_x + h)
            
            col1, col2 = st.columns([1,2])
            with col1: st.dataframe({"x": xs, "y": ys})
            with col2: st.line_chart(pd.DataFrame({"y": ys}, index=xs))
        except Exception as e: st.error(str(e))

def taylor_ui():
    st.subheader("Taylor (Orden 2)")
    f_str = st.text_input("f(x,y)", "x + y", key="taylor_f")
    st.info("Calcula y' = f, y'' = f_x + f_y*f")
    c1, c2, c3, c4 = st.columns(4)
    with c1: x0 = st.number_input("x0", 0.0, key="ty_x0")
    with c2: y0 = st.number_input("y0", 1.0, key="ty_y0")
    with c3: h = st.number_input("paso h", 0.1, key="ty_h")
    with c4: xn = st.number_input("x final", 1.0, key="ty_xn")
    
    if st.button("Calcular Taylor"):
        try:
            x, y = sp.symbols('x y')
            sym_f = parse_expr(f_str)
            # Derivada implícita para Taylor orden 2
            sym_df = sp.diff(sym_f, x) + sp.diff(sym_f, y) * sym_f
            
            f = lambdify((x,y), sym_f, 'numpy')
            df = lambdify((x,y), sym_df, 'numpy')
            
            xs, ys = [x0], [y0]
            while xs[-1] < xn - 1e-9:
                xi, yi = xs[-1], ys[-1]
                term1 = f(xi, yi)
                term2 = df(xi, yi)
                y_next = yi + h*term1 + (h**2 / 2)*term2
                xs.append(xi + h)
                ys.append(y_next)
                
            st.line_chart(pd.DataFrame({"y": ys}, index=xs))
        except Exception as e: st.error(str(e))

def rk4_ui():
    st.subheader("Runge-Kutta 4 (RK4)")
    f_str = st.text_input("f(x,y)", "x + y", key="rk4_f")
    c1, c2, c3, c4 = st.columns(4)
    with c1: x0 = st.number_input("x0", 0.0, key="rk_x0")
    with c2: y0 = st.number_input("y0", 1.0, key="rk_y0")
    with c3: h = st.number_input("paso h", 0.1, key="rk_h")
    with c4: xn = st.number_input("x final", 1.0, key="rk_xn")
    
    if st.button("Calcular RK4"):
        try:
            x, y = sp.symbols('x y')
            f = lambdify((x,y), parse_expr(f_str), 'numpy')
            xs, ys = [x0], [y0]
            while xs[-1] < xn - 1e-9:
                xi, yi = xs[-1], ys[-1]
                k1 = f(xi, yi)
                k2 = f(xi + h/2, yi + k1*h/2)
                k3 = f(xi + h/2, yi + k2*h/2)
                k4 = f(xi + h, yi + k3*h)
                y_next = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
                xs.append(xi + h)
                ys.append(y_next)
                
            col1, col2 = st.columns([1,2])
            with col1: st.dataframe({"x": xs, "y": ys})
            with col2: st.line_chart(pd.DataFrame({"y": ys}, index=xs))
        except Exception as e: st.error(str(e))
def rk2_ui():
    st.subheader("Runge-Kutta 2 (Método del Punto Medio)")
    # Mostramos la fórmula exacta de tu imagen para que no haya dudas
    st.latex(r"""
    \begin{aligned}
    k_1 &= f(t_n, y_n) \\
    k_2 &= f(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1) \\
    y_{n+1} &= y_n + h \cdot k_2
    \end{aligned}
    """)
    
    f_str = st.text_input("f(x,y)", "x + y", key="rk2_f")
    
    # Corrección: Se usa 'value=' explícitamente y se pone un min_value muy bajo o None
    c1, c2, c3, c4 = st.columns(4)
    with c1: 
        # t0 puede ser cualquier número
        x0 = st.number_input("x0 (t0)", value=0.0, format="%.2f", key="rk2_x0")
    with c2: 
        # y0 ahora permite valores menores a 1 porque definimos 'value' explícitamente
        y0 = st.number_input("y0", value=0.5, format="%.2f", key="rk2_y0")
    with c3: 
        h = st.number_input("paso h", value=0.1, step=0.01, format="%.2f", key="rk2_h")
    with c4: 
        xn = st.number_input("x final", value=0.5, step=0.1, format="%.2f", key="rk2_xn")
    
    if st.button("Calcular RK2"):
        try:
            x, y = sp.symbols('x y')
            f = lambdify((x,y), parse_expr(f_str), 'numpy')
            
            xs, ys = [x0], [y0]
            
            while xs[-1] < xn - 1e-9:
                xi, yi = xs[-1], ys[-1]
                
                # --- CAMBIOS IMPORTANTES SEGÚN TU IMAGEN ---
                
                # Paso 1: k1 (Pendiente al inicio)
                k1 = f(xi, yi)
                
                # Paso 2: k2 (Pendiente en el punto medio)
                # La imagen dice: f(tn + h/2, yn + h/2 * k1)
                k2 = f(xi + h/2, yi + (h/2) * k1)
                
                # Paso 3: Actualización
                # La imagen dice: yn+1 = yn + h * k2
                y_next = yi + h * k2
                
                # -------------------------------------------
                
                xs.append(xi + h)
                ys.append(y_next)
                
            col1, col2 = st.columns([1,2])
            with col1: 
                st.dataframe({"x": xs, "y": ys})
            with col2: 
                st.line_chart(pd.DataFrame({"y": ys}, index=xs))
                
        except Exception as e: 
            st.error(f"Error: {str(e)}")
# ==========
# MENÚ PRINCIPAL
# ==========

def main():
    st.markdown("""
    <style>
    @keyframes gradient-animation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .animated-header {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        background: linear-gradient(-45deg, #2E86C1, #2980B9, #8E44AD, #C0392B, #16A085);
        background-size: 400% 400%;
        animation: gradient-animation 10s ease infinite;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .animated-header h1 {
    font-size: 4.5em; /* Ajusta el tamaño si es necesario */
    background: linear-gradient(45deg, #FAD7A0, #F9E79F, #82E0AA, #A9CCE3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: none; /* Quitamos la sombra para un look más limpio */
    }

    .animated-header h3 {
        font-weight: 300;
        font-size: 3.5em;
        background: linear-gradient(45deg, #A9CCE3, #E8DAEF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>

    <div class="animated-header">
        <h1>Calculadora de Métodos Numéricos</h1>
        <h3>Proyecto de Modelado Computacional</h3>
    </div>
    <hr>
    """, unsafe_allow_html=True)
     # --- INICIALIZACIÓN DEL ESTADO DE SESIÓN ---
    if 'categoria' not in st.session_state:
        st.session_state.categoria = " Inicio"
    if 'metodo_nl' not in st.session_state:
        st.session_state.metodo_nl = "Bisección"
    # -------------------------------------------
    # Sidebar para navegación
    st.sidebar.header("️ Menú de Métodos")
    
    categoria = st.sidebar.radio(
        "Selecciona la categoría:",
        [" Inicio", 
        " Sistemas No Lineales", 
        " Sistemas Lineales", 
        " Álgebra Matricial",
        " Interpolación",
        " Integración",
        " Ecuaciones Diferenciales"]
    )

    if categoria == " Inicio":
        # 1. CÓDIGO CSS PARA ESTILOS Y ANIMACIONES
        st.markdown("""
        <style>
        body {
            background: linear-gradient(-45deg, #2E86C1, #2980B9, #8E44AD, #C0392B, #16A085);
            background-size: 400% 400%;
            animation: gradient-animation 10s ease infinite;
        }
        /* Animación de entrada general */
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        /* NUEVA animación de rebote para los elementos de la lista */
        @keyframes bounceIn {
            0% {
                opacity: 0;
                transform: scale(0.3) translateY(10px);
            }
            50% {
                opacity: 0.9;
                transform: scale(1.1);
            }
            80% {
                opacity: 1;
                transform: scale(0.89);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }

        .card {
            background: linear-gradient(45deg, #007bff, #0056b3); 
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
            border: 1px solid #007bff;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeIn 0.8s ease-out;
        }

        .card:hover {
            transform: translateY(-10px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.5);
        }

        .card h3, .card h4 {
            color: #FFFFFF;
            margin-top: 0;
            border-bottom: 2px solid rgba(255, 255, 255, 0.5);
            padding-bottom: 10px;
        }

        .card ul {
            list-style-type: none;
            padding-left: 0;
        }

        /* --- SECCIÓN MODIFICADA Y AÑADIDA --- */

        .card li {
            padding: 10px 15px; /* Aumentamos el padding para mejor interacción */
            color: #f0f0f0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px; /* Bordes redondeados para el hover */
            
            /* Propiedades para la animación */
            opacity: 0; /* Empiezan invisibles para la animación de entrada */
            animation: bounceIn 0.5s forwards; /* Aplicamos la animación de rebote */
            
            /* Propiedades para la interacción del cursor */
            transition: all 0.2s ease-in-out;
            cursor: pointer;
        }

        /* NUEVO: Efecto al pasar el cursor sobre un elemento de la lista */
        .card li:hover {
            color: #FFFFFF; /* Texto más brillante */
            background-color: rgba(255, 255, 255, 0.1); /* Fondo sutil que resalta */
            transform: translateX(15px) scale(1.1); /* Se mueve a la derecha */
            border-bottom-color: transparent; /* Ocultamos la línea para un look más limpio */
        }

        /* NUEVO: Retraso escalonado para la animación de entrada */
        .card li:nth-child(1) { animation-delay: 0.1s; }
        .card li:nth-child(2) { animation-delay: 0.2s; }
        .card li:nth-child(3) { animation-delay: 0.3s; }
        .card li:nth-child(4) { animation-delay: 0.4s; }
        .card li:nth-child(5) { animation-delay: 0.5s; }
        .card li:nth-child(6) { animation-delay: 0.6s; }
        .card li:nth-child(7) { animation-delay: 0.7s; }
        .card li:nth-child(8) { animation-delay: 0.8s; }
        /* Puedes añadir más si tienes listas más largas */

        .card li:last-child {
            border-bottom: none;
        }

        .welcome-text {
            text-align: center;
            animation: fadeIn 1s ease-out;
        }
        </style>
        """, unsafe_allow_html=True)

        # 2. ESTRUCTURA DE LA PÁGINA CON LAS TARJETAS DE CONTENIDO
        st.markdown("<div class='animated-header'><h1>Contenido Disponible</h1></div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        # --- CARACTERISTICAS DE LOS GIFT ---
        gif_width = 250  # Definicion de ancho de los gift
        
        col_gif1, col_gif2, col_gif3, col_gif4, col_gif5 = st.columns([1.5, 2, 2, 2, 1.5])

        with col_gif2:
            st.image(
                "https://es.symbolab.com/public/images/graphing.webp", 
                
                width=gif_width
            )
        with col_gif3:
            st.image(
                "https://es.symbolab.com/public/images/worksheets.webp", 
                
                width=gif_width
            )
        with col_gif4:
            st.image(
                "https://es.symbolab.com/public/images/calculators.webp", 
                
                width=gif_width
            )
        st.markdown("<br>", unsafe_allow_html=True) # Espacio para separar los GIFs de las tarjetas
       
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
                <h3> Sistemas No Lineales</h3>
                <ul>
                    <li>Método de Bisección</li>
                    <li>Método de la Secante</li>
                    <li>Newton-Raphson (2 variables)</li>
                    <li>Newton-Raphson (3 variables)</li>
                    <li>Newton-Raphson Modificado (2 variables)</li>
                    <li>Newton-Raphson Modificado (3 variables)</li>
                    <li>Método de Punto Fijo (2 variables)</li>
                    <li>Método de Punto Fijo (3 variables)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h3> Sistemas de Ecuaciones Lineales</h3>
                <h4>Métodos Directos:</h4>
                <ul>
                    <li>Método de la Inversa</li>
                    <li>Eliminación de Gauss</li>
                    <li>Gauss-Jordan</li>
                </ul>
                <h4>Métodos Iterativos:</h4>
                <ul>
                    <li>Jacobi</li>
                    <li>Gauss-Seidel</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <h3> Álgebra Matricial</h3>
                <ul>
                    <li>Suma de matrices</li>
                    <li>Multiplicación de matrices</li>
                    <li>Determinante</li>
                    <li>Inversa de matriz</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("""
            <div class="card" style="background: linear-gradient(135deg, #28a745, #1e7e34);">
                <h3>Interpolación</h3>
                <ul>
                    <li>Lagrange</li>
                    <li>Newton (Dif. Div.)</li>
                    <li>Mínimos Cuadrados</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown("""
            <div class="card" style="background: linear-gradient(135deg, #17a2b8, #117a8b);">
                <h3>Integración</h3>
                <ul>
                    <li>Trapecio</li>
                    <li>Simpson 1/3</li>
                    <li>Simpson 3/8</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with c6:
            st.markdown("""
            <div class="card" style="background: linear-gradient(135deg, #6610f2, #520dc2);">
                <h3>Ec. Diferenciales</h3>
                <ul>
                    <li>Euler</li>
                    <li>Taylor (Orden 2)</li>
                    <li>Runge-Kutta 4</li>
                    <li>Runge-Kutta 2</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    # SISTEMAS NO LINEALES    
    elif categoria == " Sistemas No Lineales":
        st.markdown("<h2 style='text-align:center;'>Sistemas No Lineales</h2>", unsafe_allow_html=True)
        st.markdown("---")

        metodo_nl = st.selectbox(
            "Selecciona un método:",
            ["Bisección", "Secante", "Newton-Raphson (2V)", "Newton-Raphson (3V)",
            "Newton Modificado (2V)", "Newton Modificado (3V)",
            "Punto Fijo (2V)", "Punto Fijo (3V)"]
        )
        
        if metodo_nl == "Bisección":
            metodo_biseccion_ui()
        elif metodo_nl == "Secante":
            metodo_secante_ui()
        elif metodo_nl == "Newton-Raphson (2V)":
            newton_raphson_2v_ui()
        elif metodo_nl == "Newton-Raphson (3V)":
            newton_raphson_3v_ui()
        elif metodo_nl == "Newton Modificado (2V)":
            newton_modificado_2v_ui()
        elif metodo_nl == "Newton Modificado (3V)":
            newton_modificado_3v_ui()
        elif metodo_nl == "Punto Fijo (2V)":
            punto_fijo_2v_ui()
        elif metodo_nl == "Punto Fijo (3V)":
            punto_fijo_3v_ui()

    elif categoria == " Sistemas de Ecuaciones Lineales":
        sistemas_lineales_ui()

    elif categoria == " Álgebra Matricial":
        algebra_matricial_ui()
    # --- SECCIONES NUEVAS IMPLEMENTADAS ---

    elif categoria == " Interpolación":
        st.title("Interpolación y Ajuste")
        metodo = st.selectbox("Método:", ["Polinomio de Lagrange", "Newton (Dif. Divididas)", "Mínimos Cuadrados"])
        if metodo == "Polinomio de Lagrange": polinomio_lagrange_ui()
        elif metodo == "Newton (Dif. Divididas)": newton_diferencias_divididas_ui()
        elif metodo == "Mínimos Cuadrados": minimos_cuadrados_ui()

    elif categoria == " Integración":
        st.title("Integración Numérica")
        metodo = st.selectbox("Método:", ["Trapecio", "Simpson 1/3", "Simpson 3/8"])
        if metodo == "Trapecio": integracion_trapecio_ui()
        elif metodo == "Simpson 1/3": simpson_13_ui()
        elif metodo == "Simpson 3/8": simpson_38_ui()

    elif categoria == " Ecuaciones Diferenciales":
        st.title("Ecuaciones Diferenciales (EDO)")
        metodo = st.selectbox("Método:", ["Euler", "Taylor (Orden 2)", "Runge-Kutta 4","Runge-Kutta 2"])
        if metodo == "Euler": euler_ui()
        elif metodo == "Taylor (Orden 2)": taylor_ui()
        elif metodo == "Runge-Kutta 4": rk4_ui()
        elif metodo == "Runge-Kutta 2": rk2_ui()
        
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Información")
    st.sidebar.info("""
    **Proyecto:** Métodos Numéricos

    **Curso:** Modelado Computacional

    **INTEGRANTES:**
    - Jose Antonio Vilcanqui
    - Alexander Chicalla Garcia
    - Dante Alvarez Tapia
    - Jhonel Apaza Pacompia
    - Brayhan Quispe Cama
    """)
    ##st.sidebar.success(" Aplicación lista para usar")
# Ejecutar la aplicación
if __name__ == "__main__":
    main()

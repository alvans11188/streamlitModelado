if categoria == " Inicio":
    # 1. CÓDIGO CSS PARA ESTILOS Y ANIMACIONES
    st.markdown("""
    <style>
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
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
    
    .card li {
        padding: 8px 0px;
        color: #f0f0f0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    }
    
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
    st.markdown("<div class='welcome-text'><h2>Contenido Disponible</h2></div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>📈 Sistemas No Lineales</h3>
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
            <h3>🔢 Sistemas de Ecuaciones Lineales</h3>
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
            <h3>🧮 Álgebra Matricial</h3>
            <ul>
                <li>Suma de matrices</li>
                <li>Multiplicación de matrices</li>
                <li>Determinante</li>
                <li>Inversa de matriz</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # 3. CÓDIGO DE LAS IMÁGENES (ESTE ES EL BLOQUE QUE FALTABA)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Métodos Numéricos en Acción</h3>", unsafe_allow_html=True)
    
    col_img1, col_img2, col_img3, col_img4 = st.columns(4)

    with col_img1:
        st.image(
            "images/bisection.png",
            caption="Búsqueda de raíces.",
            use_container_width=True
        )

    with col_img2:
        st.image(
            "images/gauss.png",
            caption="Sistemas lineales.",
            use_container_width=True
        )

    with col_img3:
        st.image(
            "images/matrix.png",
            caption="Álgebra matricial.",
            use_container_width=True
        )
    
    with col_img4:
        st.image(
            "images/new_image.png", # Asegúrate de que esta imagen exista
            caption="Nueva imagen.",
            use_container_width=True
        )

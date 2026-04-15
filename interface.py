import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os

# Imports do seu projeto
try:
    from ybus import build_ybus
    from newton_raphson import newton_raphson
except ImportError:
    st.error("⚠️ Certifique-se de que os arquivos .py estão na mesma pasta.")
    st.stop()

# Helpers de formatação
def formatar_vetor_latex(vec, precisao=4):
    return r"\begin{bmatrix} " + r" \\ ".join([f"{v:.{precisao}f}" for v in vec]) + r" \end{bmatrix}"

st.set_page_config(page_title="Simulador SEP Visual", layout="wide")

with st.sidebar:
    st.header("⚙️ Configurações")
    tol_input = st.number_input("Tolerância", value=0.001, format="%f")
    max_iter_input = st.number_input("Máx. Iterações", value=20)

st.markdown("<h2 style='text-align: center; color: #1976d2;'>Simulador SEP com Shunt de Barra</h2>", unsafe_allow_html=True)

html_canvas = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; background: #cfcfcf; margin: 0; user-select: none; overflow: hidden; }
        .toolbar { background: #e0e0e0; padding: 10px; display: flex; gap: 8px; border-bottom: 2px solid #999; align-items: center; }
        .btn { background: #fff; border: 1px solid #999; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; }
        #workspace { position: relative; height: 550px; background-color: #cfcfcf; }
        .component { position: absolute; cursor: pointer; display: flex; flex-direction: column; align-items: center; transform: translate(-50%, -50%); z-index: 20; }
        .barra-linha { background: #000; border-radius: 4px; transition: transform 0.2s; width: 8px; height: 120px; }
        .gerador-circulo { width: 44px; height: 44px; border: 2px solid #000; border-radius: 50%; background: #cfcfcf; display: flex; align-items: center; justify-content: center; font-weight: bold; }
        .carga-seta { width: 0; height: 0; border-left: 12px solid transparent; border-right: 12px solid transparent; border-top: 30px solid #000; }
        .label { font-size: 12px; position: absolute; white-space: nowrap; font-weight: bold; background: rgba(255,255,255,0.8); padding: 3px 6px; border-radius: 4px; pointer-events: none; }
        #wires { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 10; }
        .wire-path { stroke: #000; stroke-width: 2.5; fill: none; pointer-events: stroke; cursor: pointer; }
        .selected .barra-linha { box-shadow: 0 0 10px 3px #1976d2; background: #1976d2; }
        #properties-panel { position: absolute; right: 15px; top: 15px; width: 220px; background: #fff; border: 1px solid #999; padding: 15px; border-radius: 8px; display: none; z-index: 100; }
    </style>
</head>
<body>
    <div class="toolbar">
        <button class="btn" onclick="addBarra()">+ Barra</button>
        <button class="btn" onclick="attachComponent('gerador')">+ Gerador</button>
        <button class="btn" onclick="attachComponent('carga')">+ Carga</button>
        <button class="btn" onclick="toggleConnectMode()">🔗 Conectar</button>
        <button class="btn" onclick="deleteSelected()" style="color:red">🗑 Excluir</button>
        <button class="btn" onclick="exportarParaPython()" style="background:#2e7d32; color:white; margin-left:auto;">▶ CALCULAR</button>
    </div>
    <div id="workspace">
        <svg id="wires"></svg>
        <div id="properties-panel">
            <h4 id="panel-title">Propriedades</h4>
            <div id="panel-content"></div>
        </div>
    </div>

    <script>
        function sendMessage(data) { window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setComponentValue", value: data }, "*"); }
        window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:componentReady", apiVersion: 1 }, "*");
        setInterval(() => { window.parent.postMessage({ isStreamlitMessage: true, type: "streamlit:setFrameHeight", height: 600 }, "*"); }, 500);

        let barras = []; let linhas = []; let geradores = []; let cargas = []; let idCounter = 1;
        let selectedElement = null; let selectedType = null; let connectMode = false; let connectStartBarra = null;
        const ws = document.getElementById("workspace"); const svg = document.getElementById("wires");

        function initSlack() { const b = { id: idCounter++, type: 'slack', x: 120, y: 250, v: 1.0, theta: 0, bsh_bus: 0, rotState: 0 }; barras.push(b); renderBarra(b); }
        initSlack();

        function renderBarra(b) {
            if(b.el) b.el.remove();
            const el = document.createElement("div"); el.className = "component"; el.style.left = b.x + "px"; el.style.top = b.y + "px";
            const angle = (b.rotState || 0) * 90;
            
            // Desenho do Shunt na Barra
            let shuntHtml = "";
            if (b.bsh_bus !== 0) {
                const isCap = b.bsh_bus > 0;
                const sym = isCap ? "||" : "@@"; // Simbolismo visual simples
                shuntHtml = `<div style="position:absolute; top:70px; color:#1976d2; font-weight:bold;">${isCap ? 'Cap' : 'Ind'}: ${b.bsh_bus}</div>`;
            }

            el.innerHTML = `<div class="label" style="top:-25px">B${b.id}</div>
                            <div class="barra-linha" style="transform: rotate(${angle}deg)"></div>
                            ${shuntHtml}`;
            
            el.onmousedown = (e) => { e.stopPropagation(); if(connectMode) handleBarraConn(b); else selectElement(b, 'barra'); };
            makeDraggable(el, b); ws.appendChild(el); b.el = el;
            if(selectedElement === b) el.classList.add("selected");
        }

        function rotateBarra() {
            if (selectedType !== 'barra') return;
            selectedElement.rotState = (selectedElement.rotState + 1) % 4;
            renderBarra(selectedElement);
            updateAllWires(); // Atualiza posição do gerador/carga acoplado
        }

        function addBarra() { const b = { id: idCounter++, type: 'PQ', x: 300, y: 250, bsh_bus: 0, rotState: 0, v:1, theta:0 }; barras.push(b); renderBarra(b); }

        function positionAttached(item, barra, tipo) {
            const orbit = 100;
            let cx = barra.x, cy = barra.y;
            const state = barra.rotState || 0;
            // Lógica de rotação acoplada: o componente gira em torno da barra
            if (state === 0) { if (tipo === 'gerador') cx -= orbit; else cx += orbit; }
            else if (state === 1) { if (tipo === 'gerador') cy -= orbit; else cy += orbit; }
            else if (state === 2) { if (tipo === 'gerador') cx += orbit; else cx -= orbit; }
            else if (state === 3) { if (tipo === 'gerador') cy += orbit; else cy -= orbit; }
            
            item.el.style.left = cx + "px"; item.el.style.top = cy + "px";
            item.elWire.setAttribute("d", `M ${barra.x} ${barra.y} L ${cx} ${cy}`);
        }

        function updateAllWires() {
            linhas.forEach(l => {
                const b1 = barras.find(b => b.id === l.b1); const b2 = barras.find(b => b.id === l.b2);
                l.elPath.setAttribute("d", `M ${b1.x} ${b1.y} L ${b2.x} ${b2.y}`);
            });
            geradores.forEach(g => positionAttached(g, barras.find(b => b.id === g.barraId), 'gerador'));
            cargas.forEach(c => positionAttached(c, barras.find(b => b.id === c.barraId), 'carga'));
        }

        function handleBarraConn(b) {
            if(!connectStartBarra) { connectStartBarra = b; b.el.classList.add("selected"); }
            else if(connectStartBarra.id !== b.id) {
                const l = { id: idCounter++, b1: connectStartBarra.id, b2: b.id, r: 0.05, x: 0.2, bsh: 0 };
                const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
                path.setAttribute("class", "wire-path"); l.elPath = path;
                path.onmousedown = (e) => { e.stopPropagation(); selectElement(l, 'linha'); };
                svg.appendChild(path); linhas.push(l); updateAllWires();
                toggleConnectMode();
            }
        }

        function attachComponent(tipo) {
            if (selectedType !== 'barra' || selectedElement.type === 'slack') return;
            const b = selectedElement;
            const item = { barraId: b.id, p: 0.5, q: 0.2, v: 1.0, el: document.createElement("div"), elWire: document.createElementNS("http://www.w3.org/2000/svg", "path") };
            item.el.className = "component";
            item.el.innerHTML = tipo === 'gerador' ? `<div class="gerador-circulo">G</div>` : `<div class="carga-seta"></div>`;
            item.elWire.setAttribute("stroke", "#444"); item.elWire.setAttribute("stroke-dasharray", "4");
            svg.appendChild(item.elWire); ws.appendChild(item.el);
            if(tipo === 'gerador') { geradores.push(item); b.type = 'PV'; } else { cargas.push(item); b.type = 'PQ'; }
            renderBarra(b); updateAllWires();
        }

        function makeDraggable(el, item) {
            let isDragging = false;
            el.onmousedown = (e) => { isDragging = true; };
            window.onmousemove = (e) => {
                if (!isDragging) return;
                const rect = ws.getBoundingClientRect();
                item.x = e.clientX - rect.left; item.y = e.clientY - rect.top;
                el.style.left = item.x + "px"; el.style.top = item.y + "px"; updateAllWires();
            };
            window.onmouseup = () => isDragging = false;
        }

        function selectElement(item, type) {
            selectedElement = item; selectedType = type;
            document.querySelectorAll(".component").forEach(e => e.classList.remove("selected"));
            const panel = document.getElementById("properties-panel"); panel.style.display = "block";
            const content = document.getElementById("panel-content");
            if (type === 'barra') {
                item.el.classList.add("selected");
                content.innerHTML = `
                    <button onclick="rotateBarra()" style="width:100%; margin-bottom:10px;">↻ Girar Barra/Comp</button>
                    <label>V (pu):</label><input type="number" step="0.01" value="${item.v}" onchange="selectedElement.v=parseFloat(this.value); renderBarra(selectedElement)">
                    <label>Shunt Barra (pu):</label>
                    <input type="number" step="0.01" value="${item.bsh_bus}" onchange="selectedElement.bsh_bus=parseFloat(this.value); renderBarra(selectedElement)">
                    <p style="font-size:10px; color:gray;">(+) Capacitor, (-) Indutor</p>
                `;
            } else {
                content.innerHTML = `<label>R:</label><input type="number" step="0.01" value="${item.r}" onchange="selectedElement.r=parseFloat(this.value)">
                                     <label>X:</label><input type="number" step="0.01" value="${item.x}" onchange="selectedElement.x=parseFloat(this.value)">`;
            }
        }

        function toggleConnectMode() { connectMode = !connectMode; connectStartBarra = null; document.getElementById("properties-panel").style.display="none"; }
        function deleteSelected() { location.reload(); }

        function exportarParaPython() {
            const data = {
                barras: barras.map(b => ({
                    id: b.id, tipo: b.type, v: b.v, theta: b.theta, Bsh_bus: b.bsh_bus,
                    p: b.type==='PV' ? (geradores.find(g=>g.barraId===b.id)?.p||0) : -(cargas.find(c=>c.barraId===b.id)?.p||0),
                    q: -(cargas.find(c=>c.barraId===b.id)?.q||0)
                })),
                linhas: linhas.map(l => ({ from: l.b1, to: l.b2, R: l.r, X: l.x, Bsh: l.bsh }))
            };
            sendMessage(data);
        }
    </script>
</body>
</html>
"""

# ==========================================
# RENDERIZAÇÃO E PONTE COM O CANVAS
# ==========================================
component_dir = os.path.abspath("canvas_sep_component")

# 1. PRIMEIRO: Cria a pasta se ela não existir
if not os.path.exists(component_dir):
    os.makedirs(component_dir, exist_ok=True)

# 2. SEGUNDO: Escreve o HTML atualizado lá dentro
with open(os.path.join(component_dir, "index.html"), "w", encoding="utf-8") as f:
    f.write(html_canvas)

# 3. TERCEIRO: Agora sim, com a pasta existente, declaramos o componente
componente_canvas = components.declare_component("canvas_sep", path=component_dir)
dados_do_canvas = componente_canvas(key="meu_canvas")

if dados:
    st.markdown("---")
    id_map = {b['id']: idx + 1 for idx, b in enumerate(dados['barras'])}
    
    # 1. Matriz Admitância (O primeiro passo que você pediu)
    st.subheader("🔹 Passo 1: Matriz Admitância Nodal (Ybus)")
    with st.spinner("Calculando Ybus..."):
        # Notar que passamos dados['barras'] agora para pegar o Bsh_bus
        Ybus = build_ybus(dados['barras'], dados['linhas'])
        df_ybus = pd.DataFrame(Ybus).map(lambda c: f"{c.real:.4f} {c.imag:+.4f}j")
        st.table(df_ybus)

    # 2. Chutes Iniciais
    st.subheader("🔹 Passo 2: Chutes Iniciais")
    st.latex(r"V^{(0)} = " + formatar_vetor_latex([b['v'] for b in dados['barras']]))

    # 3. Cálculo NR
    if len(dados['linhas']) > 0:
        try:
            # Adequação para o motor
            b_list = [{"type": b['tipo'].capitalize(), "V": float(b['v']), "theta": float(b['theta']), 
                       "P": float(b['p']), "Q": float(b['q']), "Bsh_bus": float(b['Bsh_bus'])} for b in dados['barras']]
            
            V, th, logs, pvpq, pq_idx, P_spec, Q_spec = newton_raphson(b_list, Ybus, tol=tol_input, max_iter=max_iter_input)
            
            st.success("Simulação Concluída!")
            st.subheader("🔹 Resultados Finais")
            col1, col2 = st.columns(2)
            col1.metric("Tensão Final", f"{V[-1]:.4f} pu")
            col2.metric("Ângulo Final", f"{np.degrees(th[-1]):.2f} °")
            
            with st.expander("Ver Memória de Cálculo Completa"):
                st.write("**Log de Iterações**")
                st.dataframe(pd.DataFrame(logs))
        except Exception as e:
            st.error(f"Erro no cálculo: {e}")

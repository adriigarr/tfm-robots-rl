import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";
const UMBRAL_STOCK_BAJO = 10;

// Esta interfaz es un demostrador de un único almacén de referencia. No hay
// selector ni estado multi-almacén: WH02–WH05 se usan solo para evaluar la
// generalización de las políticas de navegación (ver simulacion/), no para
// esta interfaz de pedidos.
const ACTIVE_WAREHOUSE = {
  code: "WH01",
  mapId: "warehouse_map01",
};

const CATS = {
  tornilleria: { label: "Tornillería", icon: "🔩" },
  herramientas: { label: "Herramientas", icon: "🔧" },
  embalaje: { label: "Embalaje", icon: "📦" },
  epi: { label: "EPI", icon: "🦺" },
  electricidad: { label: "Electricidad", icon: "🔌" },
};

// Datos puramente demostrativos de la pantalla inicial: no proceden del
// backend ni de ninguna telemetría real del robot. Centralizados aquí para
// dejar claro que son decorado del kiosco, no estado operativo.
const KIOSK_DEMO_DATA = {
  bateriaPct: 82,
  ultimoPedido: { hora: "14:32", nombre: "Martillo", estado: "entregado" },
};

// Máquina de estados del pedido (debe coincidir con backend/models.py).
const PASOS = [
  { key: "en_cola", label: "En cola", num: "01" },
  { key: "iniciando_simulacion", label: "Iniciando", num: "02" },
  { key: "hacia_estanteria", label: "Hacia estantería", num: "03" },
  { key: "recogiendo", label: "Recogiendo", num: "04" },
  { key: "hacia_entrega", label: "Hacia entrega", num: "05" },
  { key: "entregado", label: "Entregado", num: "06" },
  { key: "retornando", label: "Retornando a base", num: "07" },
];

const ESTADO_LABEL = {
  en_cola: "En cola · esperando al robot",
  iniciando_simulacion: "Iniciando trayecto del robot",
  hacia_estanteria: "Robot en camino a la estantería",
  recogiendo: "Recogiendo producto",
  hacia_entrega: "Robot en camino al punto de entrega",
  entregado: "Entregado en el punto de entrega",
  retornando: "Robot retornando a la base",
  completado: "Pedido completado",
};

const ESTADOS_ERROR = {
  error_ejecucion: { titulo: "Error de ejecución", desc: "El robot no ha podido completar el trayecto por un error interno." },
  colision: { titulo: "Colisión detectada", desc: "El robot ha detenido el trayecto tras detectar una colisión." },
  truncado: { titulo: "Trayecto interrumpido", desc: "El trayecto se ha truncado antes de completarse." },
};

const ESTADOS_TERMINALES = ["completado", "error_ejecucion", "colision", "truncado"];

// Tras estos estados el backend bloquea el robot en "requiere_revision":
// no admite nuevos pedidos hasta que el operador confirme la revisión
// mediante POST /api/robot/reset (ver backend/store.py).
const ESTADOS_REQUIEREN_REVISION = ["colision", "truncado"];

function dosDig(n) {
  return String(n).padStart(2, "0");
}

function decorar(producto) {
  const sinStock = producto.stock <= 0;
  const status = sinStock ? "out-of-stock" : producto.stock <= UMBRAL_STOCK_BAJO ? "low-stock" : "in-stock";
  const statusLabel = { "in-stock": "En stock", "low-stock": "Stock bajo", "out-of-stock": "Sin stock" }[status];
  return {
    ...producto,
    status,
    statusLabel,
    categoriaLabel: CATS[producto.categoria].label,
    stockTxt: sinStock ? "0 ud" : `${producto.stock.toLocaleString("es-ES")} ud`,
  };
}

function Header({ pantalla, reloj, robotEstado, onInicio }) {
  const robotColor = robotEstado === "en ruta" ? "var(--safety-yellow)" : "var(--safety-green)";
  return (
    <>
      <div className="app-topbar">
        <div className="app-topbar-inner">
          <button className="app-brand" onClick={onInicio}>
            <span className="app-brand-mark" />
            Almacén {ACTIVE_WAREHOUSE.code} · Catálogo
          </button>
          <div className="app-status">
            <span className="app-robot-pill">
              <span className="app-robot-dot" style={{ background: robotColor }} />
              Robot R-01 · {robotEstado}
            </span>
            <span>{reloj}</span>
          </div>
        </div>
        <div className="ic-hazard-strip" />
      </div>
    </>
  );
}

function AlertBanner({ tone = "warning", title, message }) {
  return (
    <div className={`ic-alert ${tone === "danger" ? "ic-alert-danger" : ""}`}>
      <span className="ic-alert-ico">!</span>
      <div>
        <strong>{title}</strong>
        <span className="ic-alert-sub">{message}</span>
      </div>
    </div>
  );
}

function ProductCard({ p, onSelect }) {
  const disabled = p.status === "out-of-stock";
  return (
    <button
      className="ic-bintag"
      disabled={disabled}
      style={{ opacity: disabled ? 0.45 : 1, cursor: disabled ? "not-allowed" : "pointer" }}
      onClick={() => !disabled && onSelect(p.sku)}
    >
      <span className="ic-rivet ic-rivet-tl" />
      <span className="ic-rivet ic-rivet-tr" />
      <div className={`ic-bintag-status ic-bintag-status-${p.status}`}>
        <div className="ic-punch" />
      </div>
      <div className="ic-bintag-body">
        <div className="ic-bintag-sku">SKU · {p.sku}</div>
        <div className="ic-bintag-heading">
          <span className="ic-bintag-icon">{p.icono}</span>
          <h4 className="ic-bintag-title">{p.nombre}</h4>
        </div>
        <p className="ic-bintag-desc">{p.desc}</p>
        <div className="ic-bintag-specs">
          <span><b>UBIC.</b> {p.ubicacion}</span>
          <span><b>EXIST.</b> {p.stockTxt}</span>
        </div>
        <div className="ic-bintag-price-row">
          <span className={`ic-chip ic-chip-${p.status}`}>{p.statusLabel}</span>
          <span className="ic-bintag-cta" style={{ color: disabled ? "var(--steel-pale)" : "var(--ink)" }}>
            {disabled ? "No disponible" : "Pedir ▸"}
          </span>
        </div>
      </div>
      <div className="ic-bintag-barcode" />
      <div className="ic-bintag-barcode-num">8 4130{p.sku.slice(1)}</div>
    </button>
  );
}

function ConfirmModal({ sel, onConfirm, onCancel, enviando }) {
  return (
    <div className="modal-overlay" onClick={enviando ? undefined : onCancel}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <span>Confirmar pedido</span>
          <button className="modal-close" onClick={onCancel} aria-label="Cerrar" disabled={enviando}>✕</button>
        </div>
        <div className="modal-body">
          <div className="modal-title-row">
            <span className="modal-icon">{sel.icono}</span>
            <div>
              <div className="modal-eyebrow">SKU · {sel.sku} · {sel.categoriaLabel}</div>
              <h2 className="modal-title">{sel.nombre}</h2>
            </div>
          </div>
          <p className="modal-desc">{sel.desc}</p>
          <div className="modal-facts">
            <div>
              <div className="modal-fact-label">Ubicación de recogida</div>
              <div className="modal-fact-value">{sel.ubicacion}</div>
              <div className="modal-fact-sub">{sel.goal_id} · mapa {ACTIVE_WAREHOUSE.mapId}</div>
            </div>
            <div className="modal-facts-right">
              <span className={`ic-chip ic-chip-${sel.status}`}>{sel.statusLabel}</span>
              <span className="modal-fact-mono">{sel.stockTxt} en estantería</span>
            </div>
          </div>
          <div className="modal-actions">
            <button className="ic-btn ic-btn-primary modal-btn" onClick={onConfirm} disabled={enviando}>
              {enviando ? "Enviando pedido…" : "Confirmar pedido"}
            </button>
            <button className="ic-btn ic-btn-secondary modal-btn-sm" onClick={onCancel} disabled={enviando}>Cancelar</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [productos, setProductos] = useState([]);
  const [pantalla, setPantalla] = useState("kiosco");
  const [query, setQuery] = useState("");
  const [categoria, setCategoria] = useState("todas");
  const [stockFiltro, setStockFiltro] = useState("todos");
  const [orden, setOrden] = useState("categoria");
  const [seleccionado, setSeleccionado] = useState(null);
  const [pedido, setPedido] = useState(null);
  const [error, setError] = useState(null);
  const [enviando, setEnviando] = useState(false);
  const [reiniciando, setReiniciando] = useState(false);
  const [robotRevisado, setRobotRevisado] = useState(false);
  const [reloj, setReloj] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/products`)
      .then((res) => res.json())
      .then(setProductos)
      .catch(() => setError("No se pudo cargar el catálogo. ¿Está el backend arrancado?"));
  }, []);

  useEffect(() => {
    const tick = () => {
      const d = new Date();
      setReloj(`${dosDig(d.getHours())}:${dosDig(d.getMinutes())}`);
    };
    tick();
    const id = setInterval(tick, 20000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!pedido || ESTADOS_TERMINALES.includes(pedido.status)) return undefined;
    pollRef.current = setInterval(async () => {
      const res = await fetch(`${API_BASE}/api/orders/${pedido.order_id}`);
      const data = await res.json();
      setPedido(data);
      if (ESTADOS_TERMINALES.includes(data.status)) {
        clearInterval(pollRef.current);
        setTimeout(() => setPantalla(data.status === "completado" ? "completado" : "error"), 900);
      }
    }, 500);
    return () => clearInterval(pollRef.current);
  }, [pedido?.order_id, pedido?.status]);

  async function confirmarPedido() {
    if (enviando) return;
    const sku = seleccionado;
    setEnviando(true);
    setError(null);
    setRobotRevisado(false);
    try {
      const res = await fetch(`${API_BASE}/api/orders`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sku }),
      });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || "No se pudo crear el pedido");
      }
      const data = await res.json();
      setPedido(data);
      setSeleccionado(null);
      setPantalla("transito");
    } catch (e) {
      setError(e.message);
      setSeleccionado(null);
    } finally {
      setEnviando(false);
    }
  }

  function irAInicio() {
    setPantalla("kiosco");
    setSeleccionado(null);
  }

  function irACatalogo() {
    setPantalla("catalogo");
    setSeleccionado(null);
    setPedido(null);
    setError(null);
    setRobotRevisado(false);
  }

  async function reiniciarRobot() {
    if (reiniciando) return;
    setReiniciando(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/robot/reset`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || "No se pudo reiniciar el robot");
      }
      setRobotRevisado(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setReiniciando(false);
    }
  }

  const decorados = productos.map((p) => decorar(p));
  const bajos = decorados.filter((p) => p.status === "low-stock");
  const agotados = decorados.filter((p) => p.status === "out-of-stock");

  const filtrados = decorados.filter((p) => {
    if (categoria !== "todas" && p.categoria !== categoria) return false;
    if (stockFiltro === "en-stock" && p.status !== "in-stock") return false;
    if (stockFiltro === "bajo" && p.status !== "low-stock") return false;
    if (stockFiltro === "sin" && p.status !== "out-of-stock") return false;
    const q = query.trim().toLowerCase();
    if (q && !(p.nombre.toLowerCase().includes(q) || p.sku.toLowerCase().includes(q) || p.categoriaLabel.toLowerCase().includes(q))) {
      return false;
    }
    return true;
  });

  const ordenados = [...filtrados].sort((a, b) => {
    if (orden === "stock-desc") return b.stock - a.stock;
    if (orden === "stock-asc") return a.stock - b.stock;
    if (orden === "nombre") return a.nombre.localeCompare(b.nombre, "es");
    return 0;
  });

  const grupos = Object.keys(CATS)
    .map((k) => ({ key: k, label: CATS[k].label, items: ordenados.filter((p) => p.categoria === k) }))
    .filter((g) => g.items.length);

  const sel = seleccionado ? decorados.find((p) => p.sku === seleccionado) : null;
  const pedidoProducto = pedido ? decorados.find((p) => p.sku === pedido.sku) : null;
  const idxPaso = pedido ? PASOS.findIndex((p) => p.key === pedido.status) : -1;
  const robotEstado = pantalla === "transito" ? "en ruta" : "libre";

  return (
    <div className="app-root">
      {pantalla === "kiosco" && (
        <div className="kiosco-screen">
          <div className="app-topbar-inner kiosco-header">
            <span className="app-brand" style={{ color: "var(--white)" }}>
              <span className="app-brand-mark" />
              Almacén {ACTIVE_WAREHOUSE.code} · Catálogo
            </span>
            <span className="kiosco-clock">Kiosco 02 · {reloj}</span>
          </div>
          <div className="kiosco-hero">
            <div className="kiosco-hero-inner">
              <p className="kiosco-eyebrow"><span className="kiosco-eyebrow-line" />§ 00 · Punto de pedido</p>
              <h1 className="kiosco-title">Pide un producto<br />al robot</h1>
              <p className="kiosco-sub">Toca para abrir el catálogo del almacén. El robot recoge el artículo de su estantería y lo deja en el punto de entrega.</p>
              <button className="ic-btn ic-btn-primary kiosco-cta" onClick={irACatalogo}>Empezar</button>
            </div>
          </div>
          <div>
            <div className="ic-hazard-strip" />
            <div className="app-topbar-inner kiosco-footer">
              <span className="kiosco-footer-status">
                <span className="kiosco-status-dot" />
                Robot R-01 · libre · batería {KIOSK_DEMO_DATA.bateriaPct} %
                <span className="demo-tag">dato de demostración</span>
              </span>
              <span>
                Último pedido {KIOSK_DEMO_DATA.ultimoPedido.hora} · {KIOSK_DEMO_DATA.ultimoPedido.nombre} · {KIOSK_DEMO_DATA.ultimoPedido.estado}
                <span className="demo-tag">dato de demostración</span>
              </span>
            </div>
          </div>
        </div>
      )}

      {pantalla === "catalogo" && (
        <div className="app-shell">
          <Header pantalla={pantalla} reloj={reloj} robotEstado={robotEstado} onInicio={irAInicio} />
          <div className="app-content">
            <div className="section-eyebrow-row">
              <p className="section-eyebrow"><span className="section-eyebrow-line" />§ 01 · Catálogo · Almacén activo · {ACTIVE_WAREHOUSE.code}</p>
              <h1 className="section-title">Elige un producto</h1>
              <p className="section-sub">El robot lo recoge de su estantería y lo entrega en el punto de entrega. Los artículos sin existencias no se pueden pedir.</p>
            </div>

            {error && (
              <div style={{ marginBottom: 16 }}>
                <AlertBanner tone="danger" title="No se pudo completar el pedido" message={error} />
              </div>
            )}

            {bajos.length > 0 && (
              <div style={{ marginBottom: 24 }}>
                <AlertBanner
                  tone="warning"
                  title={`${bajos.length} SKU por debajo del umbral de ${UMBRAL_STOCK_BAJO} unidades`}
                  message={`${bajos.slice(0, 2).map((p) => p.nombre).join(", ")}${bajos.length > 2 ? ` y ${bajos.length - 2} más` : ""} necesitan orden de compra esta semana. ${agotados.length} SKU sin existencias no se pueden pedir.`}
                />
              </div>
            )}

            <div className="ic-filterbar" style={{ flexDirection: "column", alignItems: "stretch" }}>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                <input
                  type="text"
                  placeholder="BUSCAR SKU O NOMBRE"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <select value={orden} onChange={(e) => setOrden(e.target.value)}>
                  <option value="categoria">Por categoría</option>
                  <option value="stock-desc">Más existencias</option>
                  <option value="stock-asc">Menos existencias</option>
                  <option value="nombre">A – Z</option>
                </select>
              </div>

              <div className="filter-chip-row">
                <button className={`filter-chip ${categoria === "todas" ? "active" : ""}`} onClick={() => setCategoria("todas")}>
                  Todo <span className="filter-chip-count">{decorados.length}</span>
                </button>
                {Object.keys(CATS).map((k) => (
                  <button key={k} className={`filter-chip ${categoria === k ? "active" : ""}`} onClick={() => setCategoria(k)}>
                    {CATS[k].icon} {CATS[k].label} <span className="filter-chip-count">{decorados.filter((p) => p.categoria === k).length}</span>
                  </button>
                ))}
              </div>

              <div className="filter-chip-row" style={{ borderTop: "1px solid var(--border-dark)", paddingTop: 16 }}>
                <span className="filter-chip-label">Existencias</span>
                {[
                  { key: "todos", label: "Todas", count: decorados.length, dot: "var(--steel-pale)" },
                  { key: "en-stock", label: "En stock", count: decorados.filter((p) => p.status === "in-stock").length, dot: "var(--safety-green)" },
                  { key: "bajo", label: "Stock bajo", count: bajos.length, dot: "var(--safety-orange)" },
                  { key: "sin", label: "Sin stock", count: agotados.length, dot: "var(--hazard-red)" },
                ].map((s) => (
                  <button key={s.key} className={`filter-chip ${stockFiltro === s.key ? "active" : ""}`} onClick={() => setStockFiltro(s.key)}>
                    <span className="filter-chip-dot" style={{ background: s.dot }} />
                    {s.label} <span className="filter-chip-count">{s.count}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="resumen-row">
              <span>{ordenados.length === decorados.length ? `${decorados.length} SKU · ${Object.keys(CATS).length} categorías` : `${ordenados.length} de ${decorados.length} SKU`}</span>
              <span>Umbral de stock bajo · {UMBRAL_STOCK_BAJO} ud</span>
            </div>

            {ordenados.length === 0 && (
              <div className="empty-state">
                <h3>Ningún SKU coincide</h3>
                <p>Revisa la búsqueda o quita los filtros de existencias.</p>
                <button className="ic-btn ic-btn-secondary" onClick={() => { setQuery(""); setCategoria("todas"); setStockFiltro("todos"); }}>
                  Quitar filtros
                </button>
              </div>
            )}

            {orden === "categoria" && ordenados.length > 0 && grupos.map((g) => (
              <section key={g.key} className="product-section">
                <div className="product-section-header">
                  <span>{CATS[g.key].icon}</span>
                  <h2>{g.label}</h2>
                  <span className="product-section-count">{g.items.length} SKU</span>
                  <span className="product-section-rule" />
                </div>
                <div className="product-grid">
                  {g.items.map((p) => (
                    <ProductCard key={p.sku} p={p} onSelect={setSeleccionado} />
                  ))}
                </div>
              </section>
            ))}

            {orden !== "categoria" && ordenados.length > 0 && (
              <div className="product-grid">
                {ordenados.map((p) => (
                  <ProductCard key={p.sku} p={p} onSelect={setSeleccionado} />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {pantalla === "transito" && pedido && (
        <div className="transito-screen">
          <div className="transito-top">
            <span className="transito-pedido">🤖 Pedido {pedido.order_id.slice(0, 8)} · robot R-01</span>
            <span className="transito-kiosco">Kiosco 02</span>
          </div>
          <div className="transito-body">
            <div className="transito-grid">
              <div>
                <div className="transito-label">Recogiendo para ti</div>
                <h1 className="transito-title">{pedido.nombre}</h1>
                <div className="transito-sku">SKU · {pedido.sku}</div>
              </div>
              <div className="transito-bin-box">
                <div className="transito-label">Ubicación de recogida</div>
                <div className="transito-bin">{pedidoProducto ? pedidoProducto.ubicacion : ""}</div>
                <div className="transito-sku">{pedido.goal_id} · mapa {ACTIVE_WAREHOUSE.mapId}</div>
              </div>
            </div>

            <div>
              <div className="transito-steps">
                {PASOS.map((p, i) => {
                  const hecho = i < idxPaso;
                  const ahora = i === idxPaso;
                  return (
                    <div
                      key={p.key}
                      className="transito-step"
                      style={{
                        background: ahora ? "var(--gunmetal-panel)" : "transparent",
                        borderTop: `4px solid ${hecho ? "var(--safety-green)" : ahora ? "var(--safety-yellow)" : "var(--border-dark)"}`,
                      }}
                    >
                      <div className="transito-step-num" style={{ color: ahora ? "var(--safety-yellow)" : "var(--steel)" }}>{p.num}</div>
                      <div className="transito-step-label" style={{ color: ahora ? "var(--white)" : hecho ? "var(--concrete)" : "var(--steel)" }}>{p.label}</div>
                    </div>
                  );
                })}
              </div>
              <div className="transito-progress-track">
                <div className="transito-progress-fill" style={{ width: `${pedido.progress}%` }} />
              </div>
              <div className="transito-progress-labels">
                <span>{ESTADO_LABEL[pedido.status]}</span>
                <span>{pedido.progress}%</span>
              </div>
            </div>
          </div>
          <div className="transito-bottom">
            <span>No sigas al robot. Espera el aviso de entrega en el punto de entrega.</span>
          </div>
        </div>
      )}

      {pantalla === "completado" && pedido && (
        <div className="entregado-screen">
          <div className="entregado-top">
            <span className="entregado-badge">Pedido {pedido.order_id.slice(0, 8)} · completado</span>
            <span>{reloj} · Kiosco 02</span>
          </div>
          <div className="entregado-body">
            <div className="entregado-body-inner">
              <div className="entregado-heading">
                <span className="entregado-check">✓</span>
                <h1>Completado</h1>
              </div>
              <p className="entregado-sub">{pedido.nombre} se ha entregado en el punto de entrega y el robot ha vuelto a la base. Ya puedes pedir el siguiente producto.</p>
              <div className="entregado-facts">
                <div><div className="entregado-fact-label">SKU</div><div className="entregado-fact-value">{pedido.sku}</div></div>
                <div><div className="entregado-fact-label">Ubicación de recogida</div><div className="entregado-fact-value" style={{ color: "var(--safety-yellow)" }}>{pedidoProducto ? pedidoProducto.ubicacion : ""}</div></div>
                <div><div className="entregado-fact-label">Destino (goal_id)</div><div className="entregado-fact-value">{pedido.goal_id}</div></div>
              </div>
              <div className="modal-actions">
                <button className="ic-btn ic-btn-primary entregado-btn" onClick={irACatalogo}>Pedir otro producto</button>
                <button className="ic-btn entregado-btn-ghost" onClick={irAInicio}>Terminar</button>
              </div>
            </div>
          </div>
          <div className="ic-hazard-strip" />
        </div>
      )}

      {pantalla === "error" && pedido && (
        <div className="entregado-screen">
          <div className="entregado-top">
            <span style={{ color: "var(--hazard-red)" }}>Pedido {pedido.order_id.slice(0, 8)} · {pedido.status}</span>
            <span>{reloj} · Kiosco 02</span>
          </div>
          <div className="entregado-body">
            <div className="entregado-body-inner">
              <div className="entregado-heading">
                <span className="entregado-check" style={{ background: "var(--hazard-red)" }}>!</span>
                <h1>{ESTADOS_ERROR[pedido.status]?.titulo || "Pedido interrumpido"}</h1>
              </div>
              <p className="entregado-sub">
                {ESTADOS_ERROR[pedido.status]?.desc || "El trayecto no ha podido completarse."}{" "}
                {ESTADOS_REQUIEREN_REVISION.includes(pedido.status) && !robotRevisado
                  ? "El robot queda bloqueado hasta que un operador confirme la revisión."
                  : "El robot vuelve a estar disponible para un nuevo pedido."}
              </p>
              <div className="entregado-facts">
                <div><div className="entregado-fact-label">SKU</div><div className="entregado-fact-value">{pedido.sku}</div></div>
                <div><div className="entregado-fact-label">Producto</div><div className="entregado-fact-value">{pedido.nombre}</div></div>
              </div>

              {ESTADOS_REQUIEREN_REVISION.includes(pedido.status) && !robotRevisado ? (
                <>
                  <div style={{ marginBottom: 20 }}>
                    <AlertBanner
                      tone="danger"
                      title="Robot requiere revisión"
                      message="No se aceptarán nuevos pedidos hasta confirmar la revisión. Este reinicio solo desbloquea el estado lógico del prototipo: no implica una recuperación física real del robot."
                    />
                  </div>
                  {error && (
                    <div style={{ marginBottom: 20 }}>
                      <AlertBanner tone="danger" title="No se pudo reiniciar el robot" message={error} />
                    </div>
                  )}
                  <div className="modal-actions">
                    <button className="ic-btn ic-btn-primary entregado-btn" onClick={reiniciarRobot} disabled={reiniciando}>
                      {reiniciando ? "Reiniciando…" : "Confirmar revisión y reiniciar robot"}
                    </button>
                    <button className="ic-btn entregado-btn-ghost" onClick={irAInicio}>Terminar</button>
                  </div>
                </>
              ) : (
                <div className="modal-actions">
                  <button className="ic-btn ic-btn-primary entregado-btn" onClick={irACatalogo}>Volver al catálogo</button>
                  <button className="ic-btn entregado-btn-ghost" onClick={irAInicio}>Terminar</button>
                </div>
              )}
            </div>
          </div>
          <div className="ic-hazard-strip" />
        </div>
      )}

      {pantalla === "catalogo" && sel && (
        <ConfirmModal sel={sel} onConfirm={confirmarPedido} onCancel={() => setSeleccionado(null)} enviando={enviando} />
      )}
    </div>
  );
}

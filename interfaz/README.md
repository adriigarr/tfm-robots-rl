# Interfaz — catálogo de pedidos

Interfaz tipo kiosko (catálogo) para pedir productos al robot: el usuario elige
un producto, confirma, y ve el progreso del pedido hasta que se completa. La
correspondencia `sku → goal_id` es explícita y determinista (vive solo en
`backend/catalog.json`); el frontend nunca introduce ni conoce coordenadas,
rutas ni comandos.

> **Estado actual: integración con Webots implementada, pero desactivada por
> defecto.** El backend acepta dos modos de ejecución (`EXECUTION_MODE`): el
> mock (por defecto, sin tocar Webots) y uno real que lanza una simulación de
> verdad en el almacén WH01. Ningún endpoint abre Webots salvo que se active
> explícitamente `EXECUTION_MODE=webots`. Ver [Integración con
> Webots](#integración-con-webots) más abajo.

## Alcance: demostrador de un único almacén (WH01)

Este proyecto contiene cinco almacenes simulados (WH01–WH05) usados en
`simulacion/` para entrenar y evaluar políticas de navegación. **Esta
interfaz de pedidos está configurada exclusivamente como demostrador del
almacén de referencia WH01**, cuyo mapa es `warehouse_map01`. El catálogo de
`backend/catalog.json` (`"map_id": "warehouse_map01"`) y toda la
correspondencia `sku → ubicación → goal_id` pertenecen únicamente a ese mapa.

- WH02–WH05 se reservan para la evaluación experimental de generalización de
  las políticas (ver memoria del TFM); no tienen catálogo, existencias ni
  correspondencias de productos, y no se integran en esta interfaz.
- La gestión multi-almacén (selector de almacén, catálogos independientes,
  existencias por almacén) queda **fuera del alcance** de este demostrador.
  El frontend solo muestra el almacén activo como información fija — no hay
  ningún control para cambiarlo.
- Ampliar esto en el futuro a varios almacenes activos requeriría, por cada
  almacén: su propio catálogo, sus propias existencias y su propia
  correspondencia `sku → goal_id` independiente, además de un mecanismo para
  seleccionar el almacén activo (hoy inexistente a propósito).
- El identificador de almacén activo y el `map_id` están centralizados en
  `frontend/src/App.jsx::ACTIVE_WAREHOUSE` — no se repiten como cadenas
  sueltas en el resto del componente.

## Backend (FastAPI)

```
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn main:app --port 8000 --reload
```

### Modo de ejecución

El backend nunca lanza Webots automáticamente. El modo se controla con la
variable de entorno `EXECUTION_MODE`, y por defecto es `mock`:

```
EXECUTION_MODE=mock .venv/bin/uvicorn main:app --port 8000 --reload   # por defecto, no toca Webots
EXECUTION_MODE=webots .venv/bin/uvicorn main:app --port 8000          # lanza Webots de verdad para cada pedido — ver más abajo
```

`EXECUTION_MODE=webots` requiere que existan el binario de Webots, el mundo
`simulacion/worlds/warehouse_wh01_order.wbt` y el modelo PPO configurados
(ver [Integración con Webots](#integración-con-webots)); si falta alguno, el
backend falla al arrancar con un error explícito (`WebotsExecutionServiceDisabledError`)
en vez de arrancar en un estado a medias.

### Endpoints

| Método | Ruta                     | Descripción |
|--------|--------------------------|-------------|
| GET    | `/api/products`          | Devuelve el catálogo completo (incluye `ubicacion`, la posición legible de recogida). |
| POST   | `/api/orders`             | Crea un pedido. Body: `{"sku": "..."}` (cualquier otro campo, p.ej. `goal_id`, `ubicacion`, coordenadas o comandos, se rechaza con `422`). |
| GET    | `/api/orders/{order_id}` | Devuelve el estado y progreso actuales del pedido. |
| GET    | `/api/robot/status`      | Devuelve el estado operativo del robot (`disponible` / `ocupado` / `requiere_revision`). |
| POST   | `/api/robot/reset`       | Confirmación del operador tras una revisión: pasa el robot de `requiere_revision` a `disponible`. `409` si no estaba en revisión. No implica ninguna recuperación física real, solo desbloquea el estado lógico del prototipo. |

Códigos de error: `404` (producto o pedido inexistente), `409` (sin stock,
robot ocupado con otro pedido, robot en `requiere_revision`, o reset sin
revisión pendiente), `422` (payload inválido, p. ej. `goal_id` u `ubicacion`
enviados desde el cliente).

### Estados del pedido

```
en_cola → iniciando_simulacion → hacia_estanteria → recogiendo → hacia_entrega
        → entregado → retornando → completado
```

Con posibles salidas a `error_ejecucion`, `colision` o `truncado` desde
cualquier tramo del trayecto. Las transiciones entre estados están validadas
(`backend/models.py::TRANSITIONS`); un salto incoherente se rechaza.

### Estado operativo del robot

Independiente del estado de cada pedido, el robot tiene un estado operativo
propio (`backend/models.py::RobotStatus`):

- `disponible` — acepta un nuevo pedido.
- `ocupado` — tiene un pedido en curso; se mantiene ocupado desde `en_cola`
  hasta un estado terminal, incluido el tramo `entregado → retornando →
  completado` (aunque el pedido ya se haya entregado, el robot sigue ocupado
  durante el retorno simulado a la base).
- `requiere_revision` — el último pedido terminó en `colision` o `truncado`.
  En este estado `POST /api/orders` se rechaza con `409` hasta que un
  operador confirme la revisión con `POST /api/robot/reset`. `error_ejecucion`
  no exige revisión: libera el robot directamente, igual que `completado`.

### Pruebas

```
cd backend
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Cubren: catálogo (incluida `ubicacion`), creación de pedido válido, mapeo
`sku → goal_id`, rechazo de sku inexistente/sin stock/robot ocupado, rechazo
de `goal_id`/`ubicacion`/coordenadas/comandos arbitrarios desde el cliente,
transiciones válidas e inválidas de la máquina de estados, ciclo completo del
mock hasta `completado` con liberación del robot, el camino de fallo
simulado, y el bloqueo/recuperación del robot vía `requiere_revision` +
`POST /api/robot/reset` (colisión, truncamiento, rechazo de pedidos mientras
está bloqueado, recuperación tras el reset, y que `completado` no requiere
reset).

## Frontend (React + Vite)

```
cd frontend
npm install
npm run dev
```

Abre `http://localhost:5173`. El frontend espera el backend en
`http://localhost:8000` (CORS ya configurado en `backend/main.py`). El
frontend solo envía el `sku` elegido; nunca calcula ni introduce el
`goal_id`.

## Integración con Webots

`backend/execution.py::WebotsExecutionService` lanza una simulación Webots
real por pedido, con el mismo patrón que usan los scripts de evaluación de
`simulacion/` (`webots --mode=fast --no-rendering --minimize <mundo>`, un
proceso por episodio que se cierra solo al terminar). La API, el modelo de
pedido, la máquina de estados y el frontend no se han tocado para esto — es
exactamente la sustitución localizada que dejaba prevista `ExecutionService`.

**Componentes nuevos, ninguno modifica archivos existentes de entrenamiento
o evaluación:**

- `simulacion/controllers/order_fulfillment/order_fulfillment.py` — el
  controlador Webots que ejecuta el pedido. Reutiliza (sin modificarlo)
  `WebotsEnv` de `rl_train_SUB_WP_continuo/` en `stage=6` (ciclo completo:
  aproximación → recogida → entrega → retorno), forzando el episodio al
  `goal_id` del pedido en vez de un goal aleatorio. Traduce las fases
  internas del entorno (`_hacia_descarga`, `_en_retorno`, `info["colision"]`/
  `info["exito"]`/truncamiento) a los literales de `OrderStatus` y los
  escribe de forma atómica en un archivo JSON de estado.
- `simulacion/worlds/warehouse_wh01_order.wbt` — copia de
  `warehouse_1_static.wbt` (el mundo limpio de WH01, sin peatones) con el
  único cambio de `controller "rl_train_STHWP"` →
  `controller "order_fulfillment"`.
- Modelo de referencia: `rl_train_SUB_WP_continuo/pruebas/subwp_e2_1_s42_final.zip`
  (familia SUBWP, 100% de éxito sin peatones en la evaluación de
  generalización WH02–05).

**Cómo se comunican**: `WebotsExecutionService` fija variables de entorno
(`ORDER_GOAL_ID`, `ORDER_STATUS_FILE`, `ORDER_MODEL_PATH`, `ORDER_MAX_STEPS`)
antes de lanzar el proceso Webots, y sondea (`poll`, ~0.3 s) el archivo de
estado que escribe el controlador, llamando a `on_update(status, progress)`
por cada cambio — igual que hace `MockExecutionService` con sus
temporizadores, pero con datos reales. Si Webots termina sin dejar un estado
final, o se supera un timeout global (`WEBOTS_LAUNCH_TIMEOUT`, 600 s por
defecto), el pedido se marca `error_ejecucion` en vez de quedar colgado.

**Configuración** (variables de entorno, todas opcionales con valores por
defecto):

| Variable | Por defecto |
|---|---|
| `WEBOTS_BIN` | `/Applications/Webots.app/Contents/MacOS/webots` |
| `WEBOTS_WORLD_WH01` | `simulacion/worlds/warehouse_wh01_order.wbt` |
| `WEBOTS_MODEL_PATH` | `simulacion/controllers/rl_train_SUB_WP_continuo/pruebas/subwp_e2_1_s42_final.zip` |
| `WEBOTS_MAX_STEPS` | `6000` |
| `WEBOTS_STATUS_POLL_INTERVAL` | `0.3` (segundos) |
| `WEBOTS_LAUNCH_TIMEOUT` | `600` (segundos) |
| `WEBOTS_RENDER` | `0` (`1` quita `--no-rendering --minimize`, solo para depuración manual — más lento) |

**Pruebas**: `tests/test_webots_execution.py` mockea
`asyncio.create_subprocess_exec` para probar la validación de configuración
y la lógica de sondeo/traducción de estados sin lanzar Webots real — el
proyecto no ejecuta Webots en ningún test automático.

**Primer uso real**: activar `EXECUTION_MODE=webots` abre Webots de verdad
en cuanto se crea un pedido desde la interfaz. Se recomienda probarlo primero
manualmente (no como parte de una suite automática) y confirmar que no hay
ninguna evaluación de `simulacion/` en curso en la misma máquina.

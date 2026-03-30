# Notas — Connecting Deep-Reinforcement-Learning-based Obstacle Avoidance with Conventional Global Planners using Waypoint Generators

El deep reinforcement learning ha surgido como una método dinámico para evitar obstáculos en un entorno altamente dinámico.  

Los robots móviles en la industria logística se han convertido en herramientas importantes. Los sistemas de navegación típicos de estos robots hacen uso de planificadores jerárquicos. Estos consisten en:

* planificador global: calcula el camino óptimo usando un algorítmo de búsqueda como puede ser A* o Rapid Random Tree
* planificador local: ejecuta considerando las observaciones locales y los obstáculo. 

Estos sistemas pueden funcionar bien con los entornos estáticos, pero en entornos dónde existen obstáculos dinámicos este planteamiento no es muy bueno. 

El DRL (Deep Reinforcement Learning) surge como un método end to end de planificación para reemplazar los planificadores conservadores y tradicionales en entornos dinámicos. Esto se debe a la capacidad que estos métodos tienen para navegar de forma eficiente y evitar obstáculos, haciendo uso de los datos de los sensores. 

Cosas a tener en cuenta de la integración de DRL en planificadores convencionales:
* está limitada por la naturaleza miope (myopic nature) y el entrenamiento tedioso. 
* por la falta de memoria a largo plazo sestos métodos son muy sensibles al mínimo local y se puede quedar atrapado en situaciones complejas como los pasillos largos, las esquinas o caminos sin salida. 
* los esfuerzos para incrementar las capacidades  de los sistemas de DRL para conseguir navegación con larga distancia son complejos e intensifican el ya tedioso entrenamiento. 

En este trabajo se propone una entidad de interconexión llamada: <b>intermediate planner</b>. Este combina el planificador global tradicional con un plannificador local basado en DRL haciendo uso de los waypoints. 

Al contrario que en otros trabajos ya hechos, en este se busca que la generación de waypoints sea mas dinámica y flexible y que se considere tanto la información local como global. Esto consigue dos cosas:
* ofrece un horizonte al objetivo más corto
* añade una capa de seguridad en preparación para el planificador localque solo tiene acceso a los datos de los sensores. 

Se proponen dos maneras diferentes de generar waypoints y se comparan entre ellos y con los sistemas de navegación convencionales, midiendo la seguridad, la eficiencia y la robustez. 

## Trabajos Relacionados

DRL se ha estudiado de forma extensa en varias publicaciones desde la simulación, pasando por los videojuegos y hasta la navegación y se han obtenido resultados increibles en estas áreas. 

[DQN Nature Paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) es donde por primera vez se introduce la combinación del aprendizaje por refuerzo con las redes neuronales profundas. En este paper se entrena un agente para navegar en entornos desconocidos basandose unicamente en las observaciones de los sensores. 

En los años mas recientes, se ha hecho un gran uso de DRL para la planificación de caminos y la evitación de obstáculos. Los siguientes papers son algunos de los estudios que he destacado:
* [PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning](https://arxiv.org/pdf/1710.03937)
* [Learning Navigation Behaviors End-to-End with AutoRL](https://scispace.com/pdf/learning-navigation-behaviors-end-to-end-with-autorl-1zk7x2fps3.pdf)
* [Towards Deployment of Deep-Reinforcement-Learning-Based Obstacle Avoidance into Conventional Autonomous Navigation Systems](https://www.researchgate.net/publication/350749674_Towards_Deployment_of_Deep-Reinforcement-Learning-Based_Obstacle_Avoidance_into_Conventional_Autonomous_Navigation_Systems)

En los papers [Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning](https://arxiv.org/pdf/1609.07845) y [Socially Aware Motion Planning with Deep Reinforcement Learning](https://arxiv.org/pdf/1703.08862), Chen propone un método de navegación basado en DRL para espacios con mucha gente (crowded environments) y con éxito fue capaz de transferir el approach a un robot real. El agente podía contemplar y entender los comportamientos sociales y navegar de forma robusta en entornos con mucha gente. 

Everett, en el paper [Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning](https://arxiv.org/pdf/1805.01956), extendió el planteamiento haciendo uso de la arquitectura Long-Short-Term Memory y la adaptó para que fuera posible usarla con un número de obstáculos dinámicos ([Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)). 

En [Navrep Unsupervised Representations for Reinforcement Learning of Robot Navigation in Dynamic Human Environments](https://arxiv.org/pdf/2012.04406), se propone una plataforma que consiste en una simulación 2D en el que se entrenan y se prueban una variedad de algoritmos de navegación basados en DRL y se comparan entre ellos. 

Faust en el paper [PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning](https://arxiv.org/pdf/1710.03937) combina un planificador local basado en DRL con planificadores tradicionales basados en muestreo. Los mapas globales se generan por un agente DRL y después ese mismo agente es usado para ejecutar la navegación. Los investigadores consiguieron navegación de larga distancia en mapas grandes de oficinas, pero el entrenamiento era muy tedioso y requiere un gran número de parámetros y no un set up intuitivo. 

Por otro lado, Chen en los papers [Learning to set waypoints for audio visual navigation](https://arxiv.org/pdf/2008.09622) y [Soundspaces: Audio-Visual Navigation in 3D environments](https://arxiv.org/pdf/1912.11474), utiliza DRL para generar waypoints teniendo en cuenta las pistas audio visuales y una memoria acústica. Los investigadores entrenaron el agente DRL para aprender a generar waypoints óptimos en un end to end y validaron su planteamiento en mapas 3D de las tipicas residencias. 

En el paper [Combining Optimal Control and Learning for Visual Navigation in Novel Environments](https://arxiv.org/pdf/1903.02531), se entrena un modulo de percepción basado en aprendizaje para situar los waypoints en entornos desconocidos para generar un camino libre de obstáculos. El modelo se entrena usando una representación visual y se combina con un modelo basado en control convencional para la navegación. 

En el paper [Learning Local Planners for Human-aware Navigation in Indoor Environments](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9341783) se integró por primera vez un planificador local basado en DRL con un planificador global convencional y demostró resultados prometedores. Los investigadores utilizaron una muestra del camino global para crear los waypoints para el planificador local. De forma similar, en el paper [Deep Reinforcement Learning for Navigation in Cluttered Environments](https://aircconline.com/csit/papers/vol10/csit101117.pdf), se propone un subsampling o muestreo diseñado a mano para ejecutar un planificador local basado en DRL con un stack convencional de navegación. Unna limitación de estos trabajos es que el simple muestreo del camino global es inflexible y puede generar bloqueo en situaciones complejas, por ejemplo cuando los humanos estan bloquyeando el camino. 

En este trabajo se introduce un planificador intermedio para crear los waypoints de forma mas flexible teniendo en cuenta la información tanto local como global. Con esto se busca que esta creación de waypoints sea la más óptima y que se pueda hacer uso de la función de replanificación ofreciendo como resultado trayectorias mas suaves, un aumento de la eficiencia y menos choques. 

## Metodología

Para conseguir una navegación con largo alcance mientras se usa un planificador local basado en DRL, se propone una combinación con planificadores globales tradicionales. 

El diseño del sistema propuesto contiene una entidad de interconexión llamada planificador intermedio, que conecta el planificador local con el planificador global y se presentan dos maneras diferentes de generar los puntos de referencia o waypoints. Como baseline, también se propone un muestreo simple de waypoints a lo lago del camino óptimo global como se propone en los dos últimos papers mencionados previamente. Este enfoque se denomina SUB-WP. 

El segundo generador de waypoints calcla el punto de referencia usando una distancia look-ahead, o de  mirada hacia adelante, además de la posición espacial del robot. Dado un camino global como un array de poses, se muestrea un set de objetivos locales basados en la posición del robot y la distancia look-ahead. A este algoritmo se le denomina S-WP. El set de objetivos se puede ver como objetivos locales y son el input para el planificador local basado en DRL. Además, se integra también un mecanismo de replanificación para modificar el camino global cuando el robot está fuera de su curso o si el robot está parado sin movimiento durante un tiempo limitado. En esta situación, una replanificación global se activa y un nuevo objetivo local se calcula basandose en el nuevo camino global y la posición del robot. A esto se le denomina STH-WP (Spaial Time Horizon Waypoint Generator)

Utilizar un simple muestreo puede dar lugar a trayectorias poco eficientes especialmente en entornos muy dinámicos, porque la trayectoria inicial planeada se cambia de forma constante cuando se quiere evitar obstáculos. Esto da lugar a que el robot ignore caminos mas adecuados para poder llegar a los waypoints estáticos lo que resulta en trayectorias menos suaves y eficientes. A raiz de esto, se plantea el último generador de puntos de referencia en el cual se asume que no todos los puntos del camino global son igual de importantes y que realmente solo los puntos críticos son los que se tienen que visitar por el robot. A estos puntos críticos se les llama landmarks L en este paper. El waypoint generator usa estos puntos específicos pra calcular posiciones optimas para los puntos de referencia para el planificador local basado en DRL. Esto permite que las trayectorias sean más suaves y resulta en una navegación mas eficiente. 

Para seleccionar los landmarks se tiene en cuenta que en un mapa estos puntos son reconocidos por los humanos generalmente como turnong points o puntos de cambio. Es decir, para navegar por estos puntos normalmente se requiere hacer un giro y es por ello que se formula un landmark como un punto donde el angulo de giro del robot es mas grande que un valor de threshold. 
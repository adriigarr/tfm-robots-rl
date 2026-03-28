# Notas — Kästner et al. (2021)

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
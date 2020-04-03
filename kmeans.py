import numpy as np

class KMeansMyImplementation:
    
    def __init__(self, n_clusters=8, random_state=None, init='k-means++', n_init=10, 
                 max_iter=300, tol=0.0001): # параметры как в sklearn, некоторые не реализовывал
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init=init # 'k-means++', 'random' or an ndarray
        self.n_init=n_init
        self.max_iter=max_iter
        self.tol=tol # пороговое отклонение
        
    def squared_dist(self, x, y): # квадрат расстояния между двумя объектами  
        return sum((x - y)**2)
    
    def dist(self, x, y): # расстояние между двумя объектами
        return sum((x - y)**2)**0.5
    
    def norm(self, x): # норма объекта
        return sum(x**2)**0.5
    
    def crit(self, new_centr, old_centr): # Сумма расстояний от центроидов на текущем шаге
        s = 0                             # до центроидов на предыдущем. Используется как
        for x in (new_centr - old_centr): # критерий завершения поиска центроидов
            s += self.norm(x)
        return s
    
    def get_centr(self, X, oldCentr): # "центр масс" объектов из одного кластера
        if len(X):
            return sum(X)/len(X)
        else:           # если в кластере отсутствуют объекты, возвращаем предыдущий "центр масс"
            return oldCentr
        
    def get_centrs(self, X, labels, centr): # центроиды кластеров
        # M - это список, i-м элементом которого является массив, состоящий 
        # из объектов, принадлежащих i-му кластеру
        M = [np.array([X[j] for j in range(len(X)) if labels[j] == i]) for i in range(self.n_clusters)]
        return np.array([self.get_centr(M[i], centr[i]) for i in range(self.n_clusters)])
    
    def find_centr(self, n, X): # поиск объекта, наиболее удаленного от первых n объектов,
        max_d = 0               # используется в k-means++
        for x in X:
            cur_norm = self.norm(np.array([self.dist(x, X[i]) for i in range(n)]))
            if max_d < cur_norm:
                max_d = cur_norm
                max_x = x
        return max_x
    
    def fit_predict(self, X):
        
        if self.n_init <= 0:
            raise ValueError("Invalid number of initializations."
                         "n_init=%d must be bigger than zero." % self.n_init)
        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % self.max_iter)    
        if len(X) < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            len(X), self.n_clusters))
        
        
        if type(self.init) == np.ndarray:
            if self.init.shape[0] != self.n_clusters:
                raise ValueError('The shape of the initial centers (%s) '
                                 'does not match the number of clusters %i'
                                 % (self.init.shape[0], self.n_clusters))
            if self.init.shape[1] != X.shape[1]:
                raise ValueError("The number of features of the initial centers %s "
                                 "does not match the number of features of the data %s."
                                 % (self.init.shape[1], X.shape[1]))
            self.labels_ = np.array([0] * len(X))
            old_centers = np.array([[0.0] * len(X[0])] * self.n_clusters)
            self.cluster_centers_ = self.init.copy()
            eps = self.tol * self.n_clusters
            self.n_iter_ = 0
            while self.n_iter_ < self.max_iter and self.crit(self.cluster_centers_, old_centers) > eps:
                self.n_iter_ += 1
                for i in range(self.n_clusters):
                    old_centers[i] = self.cluster_centers_[i].copy()
                for i in range(len(X)):
                    min_dist = self.dist(X[i], old_centers[0])
                    jmin = 0
                    for j in range(1, self.n_clusters):
                        if self.dist(X[i], old_centers[j]) < min_dist:
                            min_dist = self.dist(X[i], old_centers[j])
                            jmin = j
                    self.labels_[i] = jmin
                self.cluster_centers_ = self.get_centrs(X, self.labels_, old_centers).copy()
            self.inertia_ = 0
            for i in range(len(X)):
                self.inertia_ += self.squared_dist(X[i], self.cluster_centers_[self.labels_[i]])
            return self.labels_
        else:
            np.random.seed(self.random_state)
            inertia = [0] * self.n_init
            labels = np.array([[0] * len(X) for l in range(self.n_init)])
            cluster_centers = np.array([[[0.0] * len(X[0])] * self.n_clusters for l in range(self.n_init)])
            eps = self.tol * self.n_clusters
            n_iter = [0] * self.n_init
            for n_init in range(self.n_init):
                old_centers = np.array([[0.0] * len(X[0])] * self.n_clusters).copy()
                if self.init == 'k-means++':
                    cluster_centers[n_init][0] = X[np.random.randint(0,len(X))].copy()
                    for i in range(1, self.n_clusters):
                        cluster_centers[n_init][i] = self.find_centr(i, X).copy()       
                elif self.init == 'random':
                    for i in range(self.n_clusters):
                        cluster_centers[n_init][i] = X[np.random.randint(0,len(X))].copy()
                else:
                    raise ValueError("the init parameter for the k-means should "
                            "be 'k-means++' or 'random' or an ndarray, "
                            "'%s' (type '%s') was passed." % (self.init, type(self.init)))
                while n_iter[n_init] < self.max_iter and self.crit(cluster_centers[n_init], old_centers) > eps:
                    n_iter[n_init] += 1
                    for i in range(self.n_clusters):
                        old_centers[i] = cluster_centers[n_init][i].copy()
                    for i in range(len(X)):
                        min_dist = self.dist(X[i], old_centers[0])
                        jmin = 0
                        for j in range(1, self.n_clusters):
                            if self.dist(X[i], old_centers[j]) < min_dist:
                                min_dist = self.dist(X[i], old_centers[j])
                                jmin = j
                        labels[n_init][i] = jmin
                    cluster_centers[n_init] = self.get_centrs(X, labels[n_init], old_centers).copy()
                for i in range(len(X)):
                    inertia[n_init] += self.squared_dist(X[i], cluster_centers[n_init][labels[n_init][i]])
                    
            best_n_init = 0
            self.inertia_ = inertia[0]
            for n_init in range(1, self.n_init):
                if inertia[n_init] < self.inertia_:
                    self.inertia_ = inertia[n_init]
                    best_n_init = n_init
            
            self.cluster_centers_ = cluster_centers[best_n_init].copy()
            self.labels_ = labels[best_n_init].copy()
            self.n_iter_ = n_iter[best_n_init]
            return self.labels_
        
    def fit(self, X):
        
        if self.n_init <= 0:
            raise ValueError("Invalid number of initializations."
                         "n_init=%d must be bigger than zero." % self.n_init)
        if self.max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % self.max_iter)    
        if len(X) < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
            len(X), self.n_clusters))
        
        if type(self.init) == np.ndarray:
            if self.init.shape[0] != self.n_clusters:
                raise ValueError('The shape of the initial centers (%s) '
                                 'does not match the number of clusters %i'
                                 % (self.init.shape[0], self.n_clusters))
            if self.init.shape[1] != X.shape[1]:
                raise ValueError("The number of features of the initial centers %s "
                                 "does not match the number of features of the data %s."
                                 % (self.init.shape[1], X.shape[1]))
            self.labels_ = np.array([0] * len(X))
            old_centers = np.array([[0.0] * len(X[0])] * self.n_clusters)
            self.cluster_centers_ = self.init.copy()
            eps = self.tol * self.n_clusters
            self.n_iter_ = 0
            while self.n_iter_ < self.max_iter and self.crit(self.cluster_centers_, old_centers) > eps:
                self.n_iter_ += 1
                for i in range(self.n_clusters):
                    old_centers[i] = self.cluster_centers_[i].copy()
                for i in range(len(X)):
                    min_dist = self.dist(X[i], old_centers[0])
                    jmin = 0
                    for j in range(1, self.n_clusters):
                        if self.dist(X[i], old_centers[j]) < min_dist:
                            min_dist = self.dist(X[i], old_centers[j])
                            jmin = j
                    self.labels_[i] = jmin
                self.cluster_centers_ = self.get_centrs(X, self.labels_, old_centers).copy()
            self.inertia_ = 0
            for i in range(len(X)):
                self.inertia_ += self.squared_dist(X[i], self.cluster_centers_[self.labels_[i]])
            return self.labels_
        else:
            np.random.seed(self.random_state)
            inertia = [0] * self.n_init
            labels = np.array([[0] * len(X) for l in range(self.n_init)])
            cluster_centers = np.array([[[0.0] * len(X[0])] * self.n_clusters for l in range(self.n_init)])
            eps = self.tol * self.n_clusters
            n_iter = [0] * self.n_init
            for n_init in range(self.n_init):
                old_centers = np.array([[0.0] * len(X[0])] * self.n_clusters).copy()
                if self.init == 'k-means++':
                    cluster_centers[n_init][0] = X[np.random.randint(0,len(X))].copy()
                    for i in range(1, self.n_clusters):
                        cluster_centers[n_init][i] = self.find_centr(i, X).copy()       
                elif self.init == 'random':
                    for i in range(self.n_clusters):
                        cluster_centers[n_init][i] = X[np.random.randint(0,len(X))].copy()
                else:
                    raise ValueError("the init parameter for the k-means should "
                            "be 'k-means++' or 'random' or an ndarray, "
                            "'%s' (type '%s') was passed." % (self.init, type(self.init)))
                while n_iter[n_init] < self.max_iter and self.crit(cluster_centers[n_init], old_centers) > eps:
                    n_iter[n_init] += 1
                    for i in range(self.n_clusters):
                        old_centers[i] = cluster_centers[n_init][i].copy()
                    for i in range(len(X)):
                        min_dist = self.dist(X[i], old_centers[0])
                        jmin = 0
                        for j in range(1, self.n_clusters):
                            if self.dist(X[i], old_centers[j]) < min_dist:
                                min_dist = self.dist(X[i], old_centers[j])
                                jmin = j
                        labels[n_init][i] = jmin
                    cluster_centers[n_init] = self.get_centrs(X, labels[n_init], old_centers).copy()
                for i in range(len(X)):
                    inertia[n_init] += self.squared_dist(X[i], cluster_centers[n_init][labels[n_init][i]])
                    
            best_n_init = 0
            self.inertia_ = inertia[0]
            for n_init in range(1, self.n_init):
                if inertia[n_init] < self.inertia_:
                    self.inertia_ = inertia[n_init]
                    best_n_init = n_init
            
            self.cluster_centers_ = cluster_centers[best_n_init].copy()
            self.labels_ = labels[best_n_init].copy()
            self.n_iter_ = n_iter[best_n_init]
        return self
        
    def predict(self, X):
        self.labels_ = np.array([0] * len(X))
        for i in range(len(X)):
            min_dist = self.dist(X[i], self.cluster_centers_[0])
            jmin = 0
            for j in range(1, self.n_clusters):
                if self.dist(X[i], self.cluster_centers_[j]) < min_dist:
                    min_dist = self.dist(X[i], self.cluster_centers_[j])
                    jmin = j
            self.labels_[i] = jmin
        return self.labels_

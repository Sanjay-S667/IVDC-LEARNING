import numpy as np

class KalmanFliter():
    def __init__(self, x0, u, mes_er, est_er, A, B, C):
        self.x = x0
        self.u = u
        self.mes_er = mes_er
        self.est_er = est_er
        self.p = np.dot(est_er, est_er.transpose())
        self.r = np.dot(mes_er, mes_er.transpose())
        self.m = 0
        print (self.r)
    def proceed(self,y):
        #Prediction step
        self.x = np.dot(A ,self.x) + np.dot(B, u)
        self.p = np.dot(A, np.dot(self.p, A.transpose()))
        print("Estimated Value : ", self.x)
        print("Estimated Covariance : ", self.p)

        #Upadting step
        self.m = np.dot(C, np.dot(self.p, C.transpose()))+ self.r
        Kk = np.dot(np.dot(self.p, C.transpose()),np.linalg.inv(self.m))
        self.diff = C@(y - self.x)
        self.x = self.x + np.dot(Kk, self.diff)
        self.p = np.dot((np.eye(2) - np.dot(Kk,C)), self.p)
        print("Predicted Value : ", self.x)
        print("Predicted Covariance : ", self.p)
        print("Kalman Gain : ", Kk)
        print("\n\n")

x = np.array([[60],[5]])
y = [np.array([[64],[5]]), np.array([[70],[7]]), np.array([[74],[9]]), np.array([[80],[11]]), np.array([[87],[13]])]
u = np.array([[2]])
mes_er = np.array([[4]])
est_er = np.array([[2],[1]])
A = np.array([[1, 1],[0, 1]])
B = np.array([[0.5],[1]])
C = np.array([[1, 0]])

KAF = KalmanFliter(x, u, mes_er, est_er, A, B, C)
for i in y:
    KAF.proceed(i)

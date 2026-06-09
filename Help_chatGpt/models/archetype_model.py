from sklearn.mixture import GaussianMixture

class ArchetypeModel:
    def __init__(self, n_archetypes=3, random_state=42):
        self.model = GaussianMixture(
            n_components=n_archetypes,
            covariance_type="full",
            random_state=random_state)
    
    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)
    
    def bic(self, X):

        return self.model.bic(X)

    def aic(self, X):

        return self.model.aic(X)

    def centers(self):

        return self.model.means_
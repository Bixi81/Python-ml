class CustomDT():
    def __init__(self, ppoints=10, max_depth=5):
        self.ppoints = ppoints
        self.max_depth = max_depth
        self.depth = 0
        self.parents = []

    def _update_splits(self, X, y):
        percentiles = np.linspace(0, 100, min(self.ppoints, X.shape[0]))
        feat_idxs = range(X.shape[1])

        potential_splits = [list(zip([feat_idx]*min(self.ppoints, X.shape[0]),
                                     np.percentile(X[:, feat_idx], percentiles)))
                            for feat_idx in feat_idxs]

        potential_splits = np.array(potential_splits).reshape(
            X.shape[1] * min(self.ppoints, X.shape[0]), 2)

        np.random.shuffle(potential_splits)
        self.splits = potential_splits

    def _find_best_split(self, X, y):
        ginis = []

        for feat_idx, split_point in self.splits:
            mask = (X[:, int(feat_idx)] >= split_point)
            y_left = y[~mask]
            y_right = y[mask]
            prob1_left = np.mean(y_left)
            prob1_right = np.mean(y_right)
            gini_left = 1 - (prob1_left**2) - ((1-prob1_left)**2)
            gini_right = 1 - (prob1_right**2) - ((1-prob1_right)**2)
            gini = ((len(y_left)*gini_left) +
                    (len(y_right)*gini_right)) / len(y)
            ginis.append(gini)

        opt_idx = np.nanargmin(ginis)
        self.gini = ginis[opt_idx]
        self.opt_feat_idx = int(self.splits[opt_idx][0])
        self.opt_split_point = self.splits[opt_idx][1]
        self.rule_left = lambda A: A[:, self.opt_feat_idx] < self.opt_split_point
        self.rule_right = lambda A: A[:, self.opt_feat_idx] >= self.opt_split_point

    def _find_best_split_lookahead(self, X, y, lookahead_depth=1):
        ginis_lookahead = []

        for feat_idx, split_point in self.splits:
            mask = (X[:, int(feat_idx)] >= split_point)
            
            if X[~mask].shape[0] > 1:
                lookahead_left = CustomDT(ppoints=self.ppoints, max_depth=lookahead_depth)
                lookahead_left.train(X[~mask], y[~mask])
                gini_left_lookahead = lookahead_left.gini
            else:
                gini_left_lookahead = 0
            
            if X[mask].shape[0] > 1:
                lookahead_right = CustomDT(ppoints=self.ppoints, max_depth=lookahead_depth)
                lookahead_right.train(X[mask], y[mask])
                gini_right_lookahead = lookahead_right.gini
            else:
                gini_right_lookahead = 1
            
            gini_lookahead = ((len(y[~mask])*gini_left_lookahead) +
                    (len(y[mask])*gini_right_lookahead)) / len(y)
            ginis_lookahead.append(gini_lookahead)
            
        opt_idx = np.nanargmin(ginis_lookahead)
        self.gini = ginis_lookahead[opt_idx]
        self.opt_feat_idx = int(self.splits[opt_idx][0])
        self.opt_split_point = self.splits[opt_idx][1]
        self.rule_left = lambda A: A[:, self.opt_feat_idx] < self.opt_split_point
        self.rule_right = lambda A: A[:, self.opt_feat_idx] >= self.opt_split_point

    def train(self, X, y, lookahead_depth=None):
        self._update_splits(X, y)
        if lookahead_depth == None:
            self._find_best_split(X, y)
        else:
            self._find_best_split_lookahead(X, y, lookahead_depth)
        mask = self.rule_right(X)
        self.left = CustomDT(ppoints=self.ppoints, max_depth=self.max_depth)
        self.left.parents = copy(self.parents)
        self.left.parents.append(self)
        self.left.prob = np.mean(y[~mask])
        self.right = CustomDT(ppoints=self.ppoints, max_depth=self.max_depth)
        self.right.parents = copy(self.parents)
        self.right.parents.append(self)
        self.right.prob = np.mean(y[mask])
        self.left.depth = self.right.depth = self.depth + 1

        if self.depth < self.max_depth - 1:
            if y[~mask].shape[0] > 1:
                self.left.train(X=X[~mask], y=y[~mask])
            if y[mask].shape[0] > 1:
                self.right.train(X=X[mask], y=y[mask])

    def _predict(self, X):
        if self.depth == 0:
            self.y_pred = []
            self.pred_mask_left = self.rule_left(X)
            self.pred_mask_right = self.rule_right(X)
            if self.max_depth == 1:
                self.y_pred.append(self.pred_mask_left * self.left.prob)
                self.y_pred.append(self.pred_mask_right * self.right.prob)
                return None
        
        if hasattr(self.left, 'rule_left'):
            self.left.pred_mask_left = self.left.rule_left(X) * self.pred_mask_left
            self.left.pred_mask_right = self.left.rule_right(X) * self.pred_mask_left
        else:
            self.parents[0].y_pred.append(self.pred_mask_left * self.left.prob)
        
        if hasattr(self.right, 'rule_left'):
            self.right.pred_mask_left = self.right.rule_left(X) * self.pred_mask_right
            self.right.pred_mask_right = self.right.rule_right(X) * self.pred_mask_right
        else:
            self.parents[0].y_pred.append(self.pred_mask_right * self.right.prob)

        if self.depth == self.max_depth - 1:
            if hasattr(self.left, 'rule_left'):
                self.parents[0].y_pred.append(self.pred_mask_left * self.left.prob)
            if hasattr(self.right, 'rule_left'):
                self.parents[0].y_pred.append(self.pred_mask_right * self.right.prob)
        
        if self.depth < self.max_depth - 1:
            if hasattr(self.left, 'rule_left'):
                self.left._predict(X)
            if hasattr(self.right, 'rule_left'):
                self.right._predict(X)
                
    def predict(self, X, prob=False):
        self._predict(X)
        self.y_pred = np.sum(self.y_pred, axis=0)
        if prob == False:
            self.y_pred = np.round(self.y_pred)
        return self.y_pred

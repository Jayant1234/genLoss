[] Implement outer loop optimizer + lookahead. 
[] Make sure the implementation is solid, by running momentum =0.9 with SGD, k=5. 
[] Grid search for SGD with momentum+lookahead, with k =[5,10], and momentum = [0.7,0.8,0.9,0.85]
[] We want the outer_optimizer =ADAM, and run the adam with default settings in this lookahead+ADAM combination, and (k=10)
[] Compare the above with ADAM baseline. (k=10)



#Main Goal: Instead of combining lookahead i.e. (seeing where k batches go with an inner optimizer), we are going to 
# use a fancy optimizer like momentum, and ADAM, after we have conducted lookahead with vanilla SGD (without momentum). 4
# that means, optimizers like ADAM and momentum, will use the slow weights to update themselves, and fast weights will simply calculate 
# normal gradients for each of the k steps and update themselves. 

# Strategy would be to change the lookahead class to include

class Lookahead(torch.optim.Optimizer):
    def __init__(self, outer_optimizer, alpha=0.5, k=5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")

        if not 0.0 <= outer_optimizer_hyperparaters < 1.0:
            raise ValueError(f"Invalid momentum: {outer_optimizer_hyperparaters}")

        self.outer_optimizer = outer_optimizer
        self.alpha = alpha
        self.k = k
        # Track number of "fast" updates so we know when to do the slow update
        self._step_count = 0

        # Copy of the fast params to “slow” buffer
        self.fast_params = []

        for group in self.outer_optimizer.param_groups:
            sp = []
            mb = []
            for p in group['params']:
                sp.append(p.clone().detach())
                mb.append(torch.zeros_like(p))
            self.fast_params.append(sp)

    @property
    def param_groups(self):
        return self.outer_optimizer.param_groups

    def zero_grad(self):
        self.outer_optimizer.zero_grad()

    def step(self, closure=None):
        """
        1. Perform one 'fast' step with the base optimizer
        2. Every k steps, update slow weights
        """
        
        self._step_count += 1
        
        # we want to find the SGD's gradients and based on that change the fast_weight we defined in the buffer. 
        # for this we will need the outer_optimizer but we don't want to update the outer_optimizer's gradients nor its states,
        #we just want the SGD's gradient and we will collect it one by one on the fast weights. 
        
        
        if self._step_count % self.k == 0:
            
            # slow update
            # we will need to update the gradients within the outer_optimizer that we calculate using interpolation. 
            
            
            for group_idx, group in enumerate(self.outer_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    #if p.grad is None:
                        #continue # this is a bug since it causes model to not update some param that may have values changed from previous gradients.
                    
                    slow = self.slow_params[group_idx][p_idx]
                    momentum = self.momentum_buffer[group_idx][p_idx]
                    
                    # Compute the difference between fast and slow weights
                    delta = p.data - slow
                    
                    p.grad.copy_(delta)
                    
            loss = self.outer_optimizer.step(closure)

        return loss
import torch
from nflows.distributions.normal import StandardNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.flows import Flow

#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/
#------------------------------------- MAF MODEL ----------------------------------------
#/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/

class MAFModel_2(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers, condition_size, num_camadas, 
    residual, rndm_mask, fun_act, drop_prob, norm_batch):
        super(MAFModel_2, self).__init__()

        self.base_distribution = StandardNormal([num_inputs])

        transforms = []
        for _ in range(num_layers):
            # transforms.append(RandomPermutation(features=num_inputs)) #Randomly permutes input dimensions to improve  
                                                                        #variable dependency modeling across layers
            transforms.append(MaskedAffineAutoregressiveTransform(features=num_inputs, hidden_features=num_hidden,
    context_features=condition_size, num_blocks=num_camadas, use_residual_blocks=residual, random_mask=rndm_mask,
    activation=fun_act, dropout_probability=drop_prob, use_batch_norm=norm_batch))

        transform = CompositeTransform(transforms)

        self.flow = Flow(transform=transform, distribution=self.base_distribution)

    def forward(self, x, condition):
        return self.flow.log_prob(x, context=condition)

    def sample(self, num_samples, condition):
        return self.flow.sample(num_samples, context=condition)

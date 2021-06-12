from KnowledgeDistillation.soft_target import SoftTarget
from KnowledgeDistillation.dml import DeepMutualLearning
from KnowledgeDistillation.ours import PearsonCorrelationLoss
OFFKD={'softtarget':SoftTarget,}
ONKD={}
ENSEMBLEKD={'dml':DeepMutualLearning,}
OURS={
    'ver1':PearsonCorrelationLoss,
}
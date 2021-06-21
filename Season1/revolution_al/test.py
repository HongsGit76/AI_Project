import cv2, random, os, sys
import numpy as np
from copy import deepcopy
from skimage import metrics
import multiprocessing as mp

filepath =sys.argv[1]
filename, ext = os.path.splitext(os.path.basename(filepath))

img = cv2.imread(filepath)
height, width, channels = img.shape


# hyperparameters
n_initial_genes = 50
n_populations = 50
prob_mutations = 0.01
prob_add = 0.3
prob_remove = 0.2

min_radius, max_radius = 5, 15
save_every_n_iter = 100

class Gene():
    def __init__(self):
        self.center = np.array([random.randint(0, width), random.randint(0, height)])
        self.radius = random.randint(min_radius, max_radius)
        self.color = np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)])
    
    def mutate(self):
        mutation_size = max(1, int(round(random.gauss(15, 4)))) / 100

        r = random.uniform(0, 1)
        if r<0.33: # radius
            self.radius = np.clip(random.randint(
                int(self.radius * (1 - mutation_size)),
                int(self.radius * (1 + mutation_size))
            ), 1, 100)
        elif r< 0.66: #center
            self.center = np.array([
                np.clip(random.randint(
                    int(self.center[0] * (1 - mutation_size)),
                    int(self.center[0] * (1 + mutation_size))
                ), 0, width),
                np.clip(random.randint(
                    int(self.center[1] * (1 - mutation_size)),
                    int(self.center[1] * (1 + mutation_size))
                ), 0, width),
            ])
        else: # color
            self.color = np.array([
                np.clip(random.randint(
                    int(self.color[0] * (1 - mutation_size)),
                    int(self.color[0] * (1 + mutation_size))
                ), 0, 255),
                np.clip(random.randint(
                    int(self.color[1] * (1 - mutation_size)),
                    int(self.color[1] * (1 + mutation_size))
                ), 0, 255),
                np.clip(random.randint(
                    int(self.color[2] * (1 - mutation_size)),
                    int(self.color[2] * (1 + mutation_size))
                ), 0, 255)
            ])

# compute fitness
def compute_fitness(genome):
    out = np.ones((height, width, channels), dtype=np.uint8) * 255

    for gene in genome:
        cv2.circle(out, center=tuple(gene.center), radius=gene.radius, color=(int(gene.color[0]),int(gene.color[1]),int(gene.color[2])), thickness=-1)

    # mean squared error
    # 두 이미지의 차이 비교
    # 생성한 이미지가 원본 이미지와 비슷하다 = mse가 낮다 = fitness가 높다
    fitness = 255. / metrics.normalized_root_mse(img, out)

    return fitness, out

#compute population
def compute_popultaion(g):
    genome = deepcopy(g)

    #mutation
    if len(genome) < 200:
        for gene in genome:
            if random.uniform(0,1)< prob_mutations:
                gene.mutate()
    
    else:
        for gene in random.sample(genome, k=int(len(genome) * prob_mutations)):
            gene.mutate()

    # add gene
    if random.uniform(0,1) < prob_add:
        genome.append(Gene())

    # remove gene
    if len(genome) > 0 and random.uniform(0, 1) < prob_remove:
        genome.remove(random.choice(genome))

    # compute fitness
    new_fitness, new_out = compute_fitness(genome)

    return new_fitness, genome, new_out

if __name__ == "__main__":
    os.makedirs('result', exist_ok=True)

    p = mp.Pool(mp.cpu_count()-1)
    
    #1st gene
    best_genome = [Gene() for _ in range(n_initial_genes)]

    best_fitness, best_out = compute_fitness(best_genome)

    n_gen = 0

    while True:
        try:
            results = p.map(compute_popultaion, [deepcopy(best_genome)]*n_populations)
        except KeyboardInterrupt:
            p.close()
            break
    
        results.append([best_fitness, best_genome, best_out])

        new_fitnesses, new_genomes, new_outs = zip(*results)

        best_result = sorted(zip(new_fitnesses, new_genomes, new_outs), key=lambda x: x[0], reverse=True)

        best_fitness, best_genome, best_out = best_result[0]

        #end of generation
        print('Generation #%s, Fitness %s' % (n_gen, best_fitness))
        n_gen += 1

        #visualize
        if n_gen % save_every_n_iter == 0:
            cv2.imwrite('result/%s_%s.jpg'%(filename, n_gen),best_out)

        cv2.imshow('best out', best_out)
        if cv2.waitKey(1) == ord('q'):
            p.close()
            break

    cv2.imshow('best out', best_out)
    cv2.waitKey(0)

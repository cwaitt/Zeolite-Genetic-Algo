#############################################################################################
### Methods to build, run, and generate children and parents for a Genetic Algorithm (GA) ###
###                                                                                       ###
### Author:         Craig Waitt                                                           ###
### Date Created:   03/04/20                                                              ###
### Date Modified:  07/09/21                                                              ###
#############################################################################################

# General ASE and Python Modules Modules
import numpy as np
from ase.io import read, write
from ase.visualize import view
from ase import Atoms
from pathlib import Path
from ase.calculators.vasp import Vasp


# GA modules
from ase.ga.utilities import closest_distances_generator, get_all_atom_types
from ase.ga.startgenerator import StartGenerator
from ase.ga.data import PrepareDB, DataConnection
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import RattleMutation, RotationalMutation, RattleRotationalMutation


# Begin Script
class GenerateParent(object):   

    """ Class to control generation of parents for GA.

    FW: Atoms object 
        Zeolite framework

    Ads: Atoms object
        Molecule to sample inside zeolite

    pop_size: int
        Number of parent structures to generate

    nads: int (default = 1)
        Number of Ads to put into a box

    mult: float (default = 1.4)
        Scaling factor to apply to ratio_of_covalent_radii to control how close Ads molecules can get to FW """

    def __init__(self,FW,Ads,pop_size,nads=1,mult=1.4):
        self.FW = FW
        self.Ads = Ads
        self.pop_size = pop_size
        self.nads = nads
        self.mult = mult

    def construct_parent(self,ads_pos = [0,0,0],cell_scale=3):

        """ creates initial parents with adsorbate box positioned at ads_pos """

        cell = self.FW.get_cell()/cell_scale

        unique_atom_types = get_all_atom_types(self.FW, self.Ads.get_atomic_numbers())
        cd = closest_distances_generator(atom_numbers = unique_atom_types,
                                         ratio_of_covalent_radii = self.mult)

        adsorb_struct = Atoms(self.Ads.get_chemical_symbols(),self.Ads.get_positions())
        blck = [(adsorb_struct,self.nads)]

        sg = StartGenerator(slab = self.FW,
                           blocks = blck,
                           blmin = cd,
                           box_to_place_in = [ads_pos,np.array(cell)])

        starting_population = [sg.get_new_candidate() for i in range(self.pop_size)]

        return starting_population

class UpdateDB(object):

    "updates the existing database with new structures and energies from file name DB"

    def __init__(self,DB):
        self.DB = DB

    def update(self):
        da = DataConnection(self.DB)
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            tg = a.get_tags()
            num = a.info['confid']
            print('Updating candidate {0}'.format(num))

            calc = Vasp(directory = './Candidates/Can-{:02d}'.format(num))
            calc.read()

            struct = calc.get_atoms()
            struct.set_tags(tg)

            da.c.update(num,
                        atoms = struct,
                        origin = 'StartingCandidateRelaxed',
                        raw_score = -1*calc.get_potential_energy(),
                        relaxed = True)

        return num


class GenerateChild(object):

    """ Uses mutation operators and comparators to generate n_to_test childer

    DB: string
        filename of database

    pop: Population class
    
    operation: OperationSelector class

    n_to_test: int
        number of children to create

    num: int
        counting variable provided the UpDateDB """

    def __init__(self, DB, pop, operation, n_to_test, num):
        self.DB = DB
        self.pop = pop
        self.operation = operation
        self.n = n_to_test
        self.num = num

    def construct_child(self):
        Operation = []
        Marriage = []

        da = DataConnection(self.DB)

        for i in range(1,self.n+1):
            print('Now starting configuration number {0}'.format(i+self.num))
            parents = self.pop.get_two_candidates()
            op = self.operation.get_operator()
            offspring, desc = op.get_new_individual(parents)

            if offspring is None:
                print('No child was created. Look into modifying operator or comparator')
                continue
            da.add_unrelaxed_candidate(offspring, description=desc)

            Marriage.append([parents[0].info['confid'],parents[1].info['confid']])
            Operation.append(desc)

        return Operation, Marriage

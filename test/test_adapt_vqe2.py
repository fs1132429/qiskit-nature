
""" Test of the Adaptive VQE ground state calculations """
import contextlib
import copy
import io
import unittest

from typing import cast

from test import QiskitNatureTestCase, requires_extra_library

import numpy as np

from qiskit.providers.basicaer import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.opflow import I,Z,X, PauliSumOp
from qiskit.opflow.gradients import Gradient, NaturalGradient
from qiskit.test import slow_test
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp


from qiskit_nature import QiskitNatureError
from qiskit_nature.algorithms import AdaptVQE, VQEUCCFactory
from qiskit_nature.adaptvqe2 import AdaptVQE2
from qiskit_nature.circuit.library import HartreeFock, UCC
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.properties.second_quantization.electronic import (
    ElectronicEnergy,
    ParticleNumber,
)
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.properties.second_quantization.electronic.bases import ElectronicBasis
from qiskit_nature.properties.second_quantization.electronic.integrals import (
    OneBodyElectronicIntegrals,
    TwoBodyElectronicIntegrals,
)


class TestAdaptVQE2(QiskitNatureTestCase):
    """Test Adaptive VQE Ground State Calculation"""

    """@requires_extra_library
    def setUp(self):
        super().setUp()

        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.735", unit=UnitsType.ANGSTROM, basis="sto3g"
        )

        self.problem = ElectronicStructureProblem(self.driver)

        self.expected = -1.85727503

        self.qubit_converter = QubitConverter(ParityMapper())

        self.inter_dist = 1.6

        self.driver1 = PySCFDriver(
            atom="Li .0 .0 .0; H .0 .0 " + str(self.inter_dist),
            unit=UnitsType.ANGSTROM,
            basis="sto3g",
        )

        self.transformer = ActiveSpaceTransformer(num_electrons=2, num_molecular_orbitals=3)

        self.problem1 = ElectronicStructureProblem(self.driver1, [self.transformer])"""
    
    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.h2_op = PauliSumOp.from_list(
            [
            ("IIII", -0.8105479805373266),
            ("ZZII",- 0.2257534922240251),
            ("IIZI",+0.12091263261776641),
            ("ZIZI", + 0.12091263261776641),
            ("IZZI", + 0.17218393261915543),
            ("IIIZ",+ 0.17218393261915546),
            ("IZIZ", + 0.1661454325638243),
            ("ZZIZ",+ 0.1661454325638243),
            ("IIZZ", - 0.2257534922240251),
            ("IZZZ",+ 0.16892753870087926),
            ("ZZZZ",+ 0.17464343068300464),
            ("IXIX",+ 0.04523279994605788),
            ("ZXIX",+ 0.04523279994605788),
            ("IXZX", - 0.04523279994605788),
            ("ZXZX", - 0.04523279994605788),
            ]
        )
        self.h2_energy = -1.85727503
        

        self.ryrz_wavefunction = TwoLocal(rotation_blocks=["ry", "rz"], entanglement_blocks="cz")
        self.ry_wavefunction = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")

        self.qasm_simulator = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=1024,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.statevector_simulator = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,)

        self.expected = -1.85727503


    """def test_default(self):
       # Default execution
        solver = VQEUCCFactory(QuantumInstance(BasicAer.get_backend("statevector_simulator")))
        calc = AdaptVQE2(self.qubit_converter, solver)
        res = calc.solve(self.problem)
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)"""

    def test_default(self):
        print("qubits:",self.h2_op.num_qubits)
        excitation_pool= [PauliSumOp(SparsePauliOp(['IIIY', 'IIZY'],
              coeffs=[ 0.5+0.j, -0.5+0.j]), coeff=1.0), PauliSumOp(SparsePauliOp(['ZYII', 'IYZI'],
              coeffs=[-0.5+0.j,  0.5+0.j]), coeff=1.0), PauliSumOp(SparsePauliOp(['IYIX', 'ZYIX', 'IYZX', 'ZYZX', 'IXIY', 'ZXIY', 'IXZY', 'ZXZY'],
              coeffs=[-0.125+0.j, -0.125+0.j,  0.125+0.j,  0.125+0.j,  0.125+0.j,  0.125+0.j, -0.125+0.j, -0.125+0.j]), coeff=1.0)]
        calc= AdaptVQE2(excitation_pool=excitation_pool,operator=self.h2_op)
        #calc= AdaptVQE3()
        res= calc.solve()
        self.assertAlmostEqual(res.electronic_energies[0], self.expected, places=6)


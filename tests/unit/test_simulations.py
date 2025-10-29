import numpy as np
import h5py
import unittest
import numpy.testing as npt

import src.initialize as initialize

class TestSimulationRegion(unittest.TestCase):
    
    def test_region_internal_variables(self):

        lower_boundary = -1
        upper_boundary = 1
        boundary_condition = 'periodic'
        TestClass = initialize.SimulationRegion(lower_boundary=lower_boundary,
                                                upper_boundary=upper_boundary,
                                                boundary_condition=boundary_condition)
        
        self.assertEqual(lower_boundary, TestClass.lower_boundary)
        self.assertEqual(upper_boundary, TestClass.upper_boundary)
        self.assertEqual(boundary_condition, TestClass.boundary_condition)
    

class TestSimulationParameters(unittest.TestCase):

    def test_parameters_internal_variables(self):
        
        num_particles = 10
        time_step = 0.1
        simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(num_particles=num_particles,
                                                    time_step=time_step,
                                                    simulation_time=simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        self.assertEqual(num_particles, TestClass.num_particles)
        self.assertEqual(time_step, TestClass.time_step)
        self.assertEqual(simulation_time, TestClass.simulation_time)
        self.assertEqual(beta, TestClass.beta)
        self.assertEqual(st, TestClass.st)
    
    def test_parameters_computed_internal_variables(self):

        num_particles = 10
        time_step = 0.5
        simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(num_particles=num_particles,
                                                    time_step=time_step,
                                                    simulation_time=simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        num_steps = 2
        time_array = np.array([0.0, 1.0])

        self.assertAlmostEqual(num_steps, TestClass.num_steps)
        npt.assert_array_almost_equal(TestClass.time_array, time_array)


class TestSimulationFlow(unittest.TestCase):

    def test_internal_variables(self):

        def flow(x, t):

            return x + t
        
        def spatial_derivative(x, t):

            return 1 + t
        
        def time_derivative(x, t):

            return x + 1
        
        TestClass = initialize.SimulationFlow(flow=flow,
                                              spatial_derivative=spatial_derivative,
                                              time_derivative=time_derivative)

        self.assertIs(flow, TestClass.flow)
        self.assertIs(spatial_derivative, TestClass.spatial_derivative)
        self.assertIs(time_derivative, TestClass.time_derivative)


class TestParticleInitialization(unittest.TestCase):
    
    def test_inital_particle_positions(self):

        lower_boundary = -1
        upper_boundary = 1
        boundary_condition = 'periodic'
        SimulationRegion = initialize.SimulationRegion(lower_boundary=lower_boundary,
                                                upper_boundary=upper_boundary,
                                                boundary_condition=boundary_condition)
        num_particles = 3
        distribution = 'uniform'

        expected_positions = np.array([-0.5, 0, 0.5])
        calculated_positions = initialize.initalize_particles(SimulationRegion=SimulationRegion,
                                                              num_particles=num_particles,
                                                              distribution=distribution)

        npt.assert_array_almost_equal(calculated_positions, expected_positions)



# Test material derivative
# Test diff_eq
# Test RK4 step
# Test particle array setup works
# Test time step arrary slicing works
# Test packaging to acc,vos,pos
# Test Saving to h5py works

if __name__ == '__main__':

    unittest.main()

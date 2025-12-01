import numpy as np
import h5py

import unittest
import numpy.testing as npt

import src.initialize as initialize
import src.simulate as simulate
import src.utils as utils

import tests.resources.expected_outputs as ex

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
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        self.assertEqual(num_particles, TestClass.num_particles)
        self.assertEqual(time_step, TestClass.time_step)
        self.assertEqual(total_simulation_time, TestClass.total_simulation_time)
        self.assertEqual(beta, TestClass.beta)
        self.assertEqual(st, TestClass.st)
    
    def test_parameters_computed_internal_variables(self):

        num_particles = 10
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        num_steps = 2
        time_array = np.array([0.0, 1.0])

        self.assertAlmostEqual(num_steps, TestClass.num_steps)
        npt.assert_array_almost_equal(TestClass.time_array, time_array)

    def test_get_current_time(self):

        num_particles = 10
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)

        expected_ans = 0
        ans = TestClass.current_time(0)

        self.assertAlmostEqual(expected_ans, ans)



class TestSimulationFlow(unittest.TestCase):

    def test_internal_variables(self):

        flow = ex.intialization_flow
        spatial_derivative = ex.intialization_spatial_derivative
        time_derivative = ex.intialization_time_derivative
        
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

class TestSaveData(unittest.TestCase):

    def test_save_extract(self):

        acc_test = np.zeros((2, 3, 1))
        pos_vel_test = np.zeros((2, 3, 2))
        pos_vel_test[:, :, 1] = 1

        save_path = 'tests/resources/extraction_test'

        utils.save_data(particle_acc=acc_test,
                        particle_pos_vel=pos_vel_test,
                        save_path=save_path)
        
        acc = utils.extract_data(save_path=save_path,
                                 extract='acceleration')
        vel = utils.extract_data(save_path=save_path,
                                 extract='velocity')
        pos = utils.extract_data(save_path=save_path,
                                 extract='position')
        
        npt.assert_equal(acc_test, acc)
        npt.assert_equal(pos_vel_test[:, :, 1], vel)
        npt.assert_equal(pos_vel_test[:, :, 0], pos)
        
    def test_extract_key(self):

        self.assertRaises(KeyError, utils.extract_data, save_path='')

class TestGenerateTrackingArrays(unittest.TestCase):

    def test_acc_array(self):

        inital_particles = np.array([-0.5, 0, 0.5])
        simulation_steps = 2
        dim_number = 1

        acc_ans, pos_vel_ans = utils.generate_tracking_arrays(initial_particles=inital_particles,
                                             simulation_steps=simulation_steps,
                                             dim_number=dim_number)

        npt.assert_equal(ex.expected_acc_ans, acc_ans)
        npt.assert_equal(ex.expected_pos_vel_ans, pos_vel_ans)
        

class TestMaterialDerivatice(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass
        particle_pos_vel = ex.simtest_particles_pos_vel
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.material_derivative(SimulationFlow=SimulationFlow,
                                                      i_particle_pos_vel=i_particle_pos_vel,
                                                      time=time)
        
        expected_ans = np.array([5, 7, 9])

        npt.assert_allclose(calculated_ans, expected_ans)


class TestDiffEq(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass
        SimulationParameters = ex.SimParamsClass
        particle_pos_vel = ex.simtest_particles_pos_vel
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.diff_eq(SimulationParameters=SimulationParameters,
                                          SimulationFlow=SimulationFlow,
                                          i_particle_pos_vel=i_particle_pos_vel,
                                          time=time)
        
        expected_ans = ex.diff_eq_expected_ans

        npt.assert_allclose(calculated_ans, expected_ans)


class TestRK4Step(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass
        SimulationParameters = ex.SimParamsClass
        particle_pos_vel = ex.simtest_particles_pos_vel
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.RK4_step(SimulationParameters=SimulationParameters,
                                           SimulationFlow=SimulationFlow,
                                           i_particle_pos_vel=i_particle_pos_vel,
                                           time=time)
        
        expected_ans = ex.rk4_expected_ans
        
        npt.assert_allclose(expected_ans, calculated_ans)


class TestRunSimulation(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass
        SimulationParameters = ex.SimParamsClass
        SimulationRegion = ex.SimRegionClass
        initial_particles = ex.simtest_initial_particles
        save_path = 'tests/resources/run_simulation_test'


        saved_ans = simulate.run_simulation(SimulationRegion=SimulationRegion,
                                                 SimulationParameters=SimulationParameters,
                                                 SimulationFlow=SimulationFlow,
                                                 initial_particles=initial_particles,
                                                 save_path=save_path)
        
        acc = utils.extract_data(save_path=save_path, extract='acceleration')
        vel = utils.extract_data(save_path=save_path, extract='velocity')
        pos = utils.extract_data(save_path=save_path, extract='position')
        
        expected_acc_ans = ex.run_simulation_expected_acc_ans
        
        expected_vel_ans = ex.run_simulation_expected_vel_ans
        
        expected_pos_ans = ex.run_simulation_expected_pos_ans

        npt.assert_allclose(expected_acc_ans, acc, rtol=1e-2)
        npt.assert_allclose(expected_vel_ans, vel, rtol=1e-2)
        npt.assert_allclose(expected_pos_ans, pos, rtol=1e-2)
        
if __name__ == '__main__':

    unittest.main()

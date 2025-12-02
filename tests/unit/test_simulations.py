import numpy as np
import h5py

import unittest
import numpy.testing as npt

import src.initialize as initialize
import src.simulate as simulate
import src.utils as utils
import src.boundary_conditions as bc

import tests.resources.expected_outputs as ex

class TestSimulationRegion(unittest.TestCase):
    
    def test_region_internal_variables(self):
        
        dim_number = 1
        lower_boundaries = [-1]
        upper_boundaries = [1]
        boundary_conditions = ['periodic']
        TestClass = initialize.SimulationRegion(dim_number=dim_number,
                                                lower_boundaries=lower_boundaries,
                                                upper_boundaries=upper_boundaries,
                                                boundary_conditions=boundary_conditions)
        
        self.assertEqual(dim_number, TestClass.dim_number)
        self.assertEqual(lower_boundaries, TestClass.lower_boundaries)
        self.assertEqual(upper_boundaries, TestClass.upper_boundaries)
        self.assertEqual(boundary_conditions, TestClass.boundary_conditions)

    def test_raises(self):

        dim_number = 2
        lower_boundaries = [-1, 1]
        upper_boundaries = [1]
        boundary_conditions = ['periodic', 'periodic']
    
        self.assertRaises(ValueError, initialize.SimulationRegion, dim_number=dim_number,
                                                lower_boundaries=lower_boundaries,
                                                upper_boundaries=upper_boundaries,
                                                boundary_conditions=boundary_conditions)
        
        dim_number = 2
        lower_boundaries = [-1, -1]
        upper_boundaries = [1, 1]
        boundary_conditions = ['periodic']
    
        self.assertRaises(ValueError, initialize.SimulationRegion, dim_number=dim_number,
                                                lower_boundaries=lower_boundaries,
                                                upper_boundaries=upper_boundaries,
                                                boundary_conditions=boundary_conditions)
    


class TestSimulationParameters(unittest.TestCase):

    def test_parameters_internal_variables(self):
        
        dim_number = 1
        num_particles = 10
        time_step = 0.1
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        self.assertEqual(dim_number, TestClass.dim_number)
        self.assertEqual(num_particles, TestClass.num_particles)
        self.assertEqual(time_step, TestClass.time_step)
        self.assertEqual(total_simulation_time, TestClass.total_simulation_time)
        self.assertEqual(beta, TestClass.beta)
        self.assertEqual(st, TestClass.st)
    
    def test_parameters_computed_internal_variables(self):

        dim_number = 1
        num_particles = 10
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        num_steps = 2
        time_array = np.array([0.0, 1.0])

        self.assertAlmostEqual(num_steps, TestClass.num_steps)
        npt.assert_array_almost_equal(TestClass.time_array, time_array)

    def test_get_current_time(self):

        dim_number = 1
        num_particles = 10
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        TestClass = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)

        expected_ans = 0
        ans = TestClass.current_time(0)

        self.assertAlmostEqual(expected_ans, ans)



class TestSimulationFlow(unittest.TestCase):

    def test_internal_variables(self):

        dim_number = 1
        flow = ex.intialization_flow
        jacobian = ex.intialization_spatial_derivative
        time_derivative = [ex.intialization_time_derivative]
        
        TestClass = initialize.SimulationFlow(dim_number=dim_number,
                                              flow=flow,
                                              jacobian=jacobian,
                                              time_derivative=time_derivative)

        self.assertEqual(dim_number, TestClass.dim_number)
        self.assertIs(flow, TestClass.flow)
        self.assertIs(jacobian, TestClass.jacobian)
        self.assertIs(time_derivative, TestClass.time_derivative)


class TestParticleInitialization(unittest.TestCase):
    
    def test_inital_particle_positions_1d(self):

        dim_number = 1
        num_particles = 3
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        SimulationParameters = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        lower_boundaries = [-1]
        upper_boundaries = [1]
        boundary_conditions = ['open']
        SimulationRegion = initialize.SimulationRegion(dim_number=dim_number,
                                                       lower_boundaries=lower_boundaries,
                                                       upper_boundaries=upper_boundaries,
                                                       boundary_conditions=boundary_conditions)

        distribution = 'uniform'

        expected_positions = np.array([[-0.5],
                                       [0],
                                       [0.5]])
        
        calculated_positions = initialize.initalize_particles(SimulationRegion=SimulationRegion,
                                                              SimulationParameters=SimulationParameters,
                                                              distribution=distribution)

        npt.assert_allclose(calculated_positions, expected_positions)

    def test_inital_particle_positions_2d(self):

        dim_number = 2
        num_particles = 4
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        SimulationParameters = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        
        lower_boundaries = [-1, -1]
        upper_boundaries = [1, 1]
        boundary_conditions = ['open', 'open']
        SimulationRegion = initialize.SimulationRegion(dim_number=dim_number,
                                                       lower_boundaries=lower_boundaries,
                                                       upper_boundaries=upper_boundaries,
                                                       boundary_conditions=boundary_conditions)

        distribution = 'uniform'

        expected_positions = ex.expected_2d_initial_particles
        
        calculated_positions = initialize.initalize_particles(SimulationRegion=SimulationRegion,
                                                              SimulationParameters=SimulationParameters,
                                                              distribution=distribution)

        npt.assert_allclose(calculated_positions, expected_positions, rtol=1e-1)

    def test_inital_particle_positions_3d(self):

        dim_number = 3
        num_particles = 8
        time_step = 0.5
        total_simulation_time = 1
        beta = 1
        st = 1
        SimulationParameters = initialize.SimulationParameters(dim_number=dim_number,
                                                    num_particles=num_particles,
                                                    time_step=time_step,
                                                    total_simulation_time=total_simulation_time,
                                                    beta=beta,
                                                    st=st)
        lower_boundaries = [-1, -1 ,-1]
        upper_boundaries = [1, 1, 1]
        boundary_conditions = ['open', 'open','open']
        SimulationRegion = initialize.SimulationRegion(dim_number=dim_number,
                                                       lower_boundaries=lower_boundaries,
                                                       upper_boundaries=upper_boundaries,
                                                       boundary_conditions=boundary_conditions)

        distribution = 'uniform'

        expected_positions = ex.expected_3d_initial_particles
        
        calculated_positions = initialize.initalize_particles(SimulationRegion=SimulationRegion,
                                                              SimulationParameters=SimulationParameters,
                                                              distribution=distribution)
        
        npt.assert_allclose(calculated_positions, expected_positions, rtol=1e-1)

class TestSaveData(unittest.TestCase):

    def test_save_extract(self):

        acc_test = np.zeros((2, 3, 1))
        pos_vel_test = np.zeros((2, 3, 2))
        pos_vel_test[:, :, 1] = 1
        dim_number = 1

        save_path = 'tests/resources/extraction_test'

        utils.save_data(particle_acc=acc_test,
                        particle_pos_vel=pos_vel_test,
                        dim_number=dim_number,
                        save_path=save_path)
        
        acc = utils.extract_data(save_path=save_path,
                                 extract='acceleration')
        vel = utils.extract_data(save_path=save_path,
                                 extract='velocity')
        pos = utils.extract_data(save_path=save_path,
                                 extract='position')

        npt.assert_equal(acc_test, acc)
        npt.assert_equal(pos_vel_test[:, :, 1:2], vel)
        npt.assert_equal(pos_vel_test[:, :, 0:1], pos)
        
    def test_extract_raises(self):

        self.assertRaises(KeyError, utils.extract_data, save_path='')

class TestGenerateTrackingArrays(unittest.TestCase):

    def test_correct_1d(self):

        inital_particles = np.array([[-0.5],
                                     [0],
                                     [0.5]])
        simulation_steps = 2
        dim_number = 1

        acc_ans, pos_vel_ans = utils.generate_tracking_arrays(initial_particles=inital_particles,
                                             simulation_steps=simulation_steps,
                                             dim_number=dim_number)

        npt.assert_equal(ex.expected_acc_ans, acc_ans)
        npt.assert_equal(ex.expected_pos_vel_ans, pos_vel_ans)
    
    def test_shape_2d(self):

        inital_particles = np.array([[-0.5, 1],
                                     [0, 2],
                                     [0.5, 3]])
        simulation_steps = 2
        dim_number = 2

        acc_ans, pos_vel_ans = utils.generate_tracking_arrays(initial_particles=inital_particles,
                                             simulation_steps=simulation_steps,
                                             dim_number=dim_number)
        
        npt.assert_equal(pos_vel_ans.shape, (3, 2, 4))
    
    def test_shape_3d(self):

        inital_particles = np.array([[-0.5, 1, 4],
                                     [0, 2, 5],
                                     [0.5, 3, 6]])
        simulation_steps = 2
        dim_number = 3

        acc_ans, pos_vel_ans = utils.generate_tracking_arrays(initial_particles=inital_particles,
                                             simulation_steps=simulation_steps,
                                             dim_number=dim_number)
        
        npt.assert_equal(pos_vel_ans.shape, (3, 2, 6))
        

class TestMaterialDerivatice(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.material_derivative(SimulationFlow=SimulationFlow,
                                                      i_particle_pos_vel=i_particle_pos_vel,
                                                      time=time)
        
        expected_ans = np.array([[2.5],
                                 [3],
                                 [3.5]])

        npt.assert_allclose(calculated_ans, expected_ans)
    
    def test_shape_1d(self):

        SimulationFlow = ex.SimFlowClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.material_derivative(SimulationFlow=SimulationFlow,
                                                      i_particle_pos_vel=i_particle_pos_vel,
                                                      time=time)

        npt.assert_equal(calculated_ans.shape, (3, 1))

    def test_shape_2d(self):

        SimulationFlow = ex.SimFlowClass2d
        particle_pos_vel = ex.simtest_particles_pos_vel_2d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.material_derivative(SimulationFlow=SimulationFlow,
                                                      i_particle_pos_vel=i_particle_pos_vel,
                                                      time=time)

        npt.assert_equal(calculated_ans.shape, (4, 2))
    
    def test_shape_3d(self):

        SimulationFlow = ex.SimFlowClass3d
        particle_pos_vel = ex.simtest_particles_pos_vel_3d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.material_derivative(SimulationFlow=SimulationFlow,
                                                      i_particle_pos_vel=i_particle_pos_vel,
                                                      time=time)

        npt.assert_equal(calculated_ans.shape, (8, 3))
        
class TestDiffEq(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass1d
        SimulationParameters = ex.SimParamsClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.diff_eq(SimulationParameters=SimulationParameters,
                                          SimulationFlow=SimulationFlow,
                                          i_particle_pos_vel=i_particle_pos_vel,
                                          time=time)
        
        expected_ans = ex.diff_eq_expected_ans_1d

        npt.assert_allclose(calculated_ans, expected_ans)

    def test_shape_2d(self):

        SimulationFlow = ex.SimFlowClass1d
        SimulationParameters = ex.SimParamsClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.diff_eq(SimulationParameters=SimulationParameters,
                                          SimulationFlow=SimulationFlow,
                                          i_particle_pos_vel=i_particle_pos_vel,
                                          time=time)

        npt.assert_equal(calculated_ans.shape, (3, 2))

    def test_shape_2d(self):

        SimulationFlow = ex.SimFlowClass2d
        SimulationParameters = ex.SimParamsClass2d
        particle_pos_vel = ex.simtest_particles_pos_vel_2d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.diff_eq(SimulationParameters=SimulationParameters,
                                          SimulationFlow=SimulationFlow,
                                          i_particle_pos_vel=i_particle_pos_vel,
                                          time=time)

        npt.assert_equal(calculated_ans.shape, (4, 4))
    
    def test_shape_3d(self):

        SimulationFlow = ex.SimFlowClass3d
        SimulationParameters = ex.SimParamsClass3d
        particle_pos_vel = ex.simtest_particles_pos_vel_3d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.diff_eq(SimulationParameters=SimulationParameters,
                                          SimulationFlow=SimulationFlow,
                                          i_particle_pos_vel=i_particle_pos_vel,
                                          time=time)

        npt.assert_equal(calculated_ans.shape, (8, 6))
    
        
        


class TestRK4Step(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass1d
        SimulationParameters = ex.SimParamsClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.RK4_step(SimulationParameters=SimulationParameters,
                                           SimulationFlow=SimulationFlow,
                                           i_particle_pos_vel=i_particle_pos_vel,
                                           time=time)
        
        expected_ans = ex.rk4_expected_ans_1d
        
        npt.assert_allclose(expected_ans, calculated_ans, rtol=1e-4)
    
    def test_shape_1d(self):

        SimulationFlow = ex.SimFlowClass1d
        SimulationParameters = ex.SimParamsClass1d
        particle_pos_vel = ex.simtest_particles_pos_vel_1d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.RK4_step(SimulationParameters=SimulationParameters,
                                           SimulationFlow=SimulationFlow,
                                           i_particle_pos_vel=i_particle_pos_vel,
                                           time=time)

        npt.assert_equal(calculated_ans.shape, (3, 2))

    def test_shape_2d(self):

        SimulationFlow = ex.SimFlowClass2d
        SimulationParameters = ex.SimParamsClass2d
        particle_pos_vel = ex.simtest_particles_pos_vel_2d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.RK4_step(SimulationParameters=SimulationParameters,
                                           SimulationFlow=SimulationFlow,
                                           i_particle_pos_vel=i_particle_pos_vel,
                                           time=time)

        npt.assert_equal(calculated_ans.shape, (4, 4))
    
    def test_shape_3d(self):

        SimulationFlow = ex.SimFlowClass3d
        SimulationParameters = ex.SimParamsClass3d
        particle_pos_vel = ex.simtest_particles_pos_vel_3d
        i_particle_pos_vel = particle_pos_vel[:, 0]
        time = 2

        calculated_ans = simulate.RK4_step(SimulationParameters=SimulationParameters,
                                           SimulationFlow=SimulationFlow,
                                           i_particle_pos_vel=i_particle_pos_vel,
                                           time=time)

        npt.assert_equal(calculated_ans.shape, (8, 6))


class TestRunSimulation(unittest.TestCase):

    def test_correct_ans(self):

        SimulationFlow = ex.SimFlowClass1d
        SimulationParameters = ex.SimParamsClass1d
        SimulationRegion = ex.SimRegionClass1d
        initial_particles = ex.simtest_initial_particles_1d
        save_path = 'tests/resources/run_simulation_test'


        saved_ans = simulate.run_simulation(SimulationRegion=SimulationRegion,
                                                 SimulationParameters=SimulationParameters,
                                                 SimulationFlow=SimulationFlow,
                                                 initial_particles=initial_particles,
                                                 save_path=save_path)
        
        acc = utils.extract_data(save_path=save_path, extract='acceleration')
        vel = utils.extract_data(save_path=save_path, extract='velocity')
        pos = utils.extract_data(save_path=save_path, extract='position')

        expected_acc_ans = ex.run_simulation_expected_acc_ans_1d
        
        expected_vel_ans = ex.run_simulation_expected_vel_ans_1d
        
        expected_pos_ans = ex.run_simulation_expected_pos_ans_1d

        npt.assert_allclose(expected_acc_ans, acc, rtol=1e-2)
        npt.assert_allclose(expected_vel_ans, vel, rtol=1e-2)
        npt.assert_allclose(expected_pos_ans, pos, rtol=1e-2)


class TestCreateFlowFuncs(unittest.TestCase):

    def test_create_flow_func(self):

        flow_list_1d = ['x**2 + t']
        _, flow_1d, _, _ = utils.create_flow_functions(flow_list=flow_list_1d)
        expected_ans_flow_1d = np.array([5])
        flow_1d_ans = flow_1d(np.array([2]), t=1)
        npt.assert_allclose(expected_ans_flow_1d, flow_1d_ans)

        flow_list_2d = ['x**2 + t', 'y**2']
        _, flow_2d, _, _ = utils.create_flow_functions(flow_list=flow_list_2d)
        expected_ans_flow_2d = np.array([5, 9])
        flow_2d_ans = flow_2d(np.array([2, 3]), t=1)
        npt.assert_allclose(expected_ans_flow_2d, flow_2d_ans)

        flow_list_3d = ['x**2 + t', 'y**2', 'z + 2*t']
        _, flow_3d, _, _ = utils.create_flow_functions(flow_list=flow_list_3d)
        expected_ans_flow_3d = np.array([5, 9, 6])
        flow_3d_ans = flow_3d(np.array([2, 3, 4]), t=1)
        npt.assert_allclose(expected_ans_flow_3d, flow_3d_ans)
    
    def test_many_particles(self):

        flow_list_2d = ['x**2 + t', 'y**2']
        _, flow_2d, _, _ = utils.create_flow_functions(flow_list=flow_list_2d)
        expected_ans_flow_2d = np.array([[5, 9], [5, 9]])
        flow_2d_ans = flow_2d(np.array([[2, 3],[2, 3]]), t=1)
        npt.assert_allclose(expected_ans_flow_2d, flow_2d_ans)
    
    def test_time_derivative(self):

        flow_list_2d = ['x**2 + 3*t', 'y**2 + t**2']
        _, _, _, dFlowdt = utils.create_flow_functions(flow_list=flow_list_2d)
        dt_expected_ans_flow_2d = np.array([3, 2])
        dt_flow_2d_ans = dFlowdt(np.array([2, 3]), t=1)
        npt.assert_allclose(dt_expected_ans_flow_2d, dt_flow_2d_ans)
    
    def test_jacobian(self):

        flow_list_2d = ['x**2 + 3*t', 'x*y**2 + t**2']
        _, _, dFlowdx, _ = utils.create_flow_functions(flow_list=flow_list_2d)
        dx_expected_ans_flow_2d = np.array([[4, 0], [9, 12]])
        dx_flow_2d_ans = dFlowdx(np.array([2, 3]), t=1)
        npt.assert_allclose(dx_expected_ans_flow_2d, dx_flow_2d_ans)

class TestBoundaryConditions(unittest.TestCase):

    def test_open_bc(self):

        i1_particle_pos_vel = np.zeros((2,1,2))

        ans = bc.open_bc(i1_particle_pos_vel=i1_particle_pos_vel)

        npt.assert_allclose(i1_particle_pos_vel, ans)
    
    def test_periodic_bc(self):

        dim_number = 1
        lower_boundaries = [0]
        upper_boundaries = [1]
        boundary_conditions = ['periodic']
        TestClass = initialize.SimulationRegion(dim_number=dim_number,
                                                lower_boundaries=lower_boundaries,
                                                upper_boundaries=upper_boundaries,
                                                boundary_conditions=boundary_conditions)
        
        i1_particle_pos_vel = np.zeros((3,1))
        i1_particle_pos_vel[0] = 0.5
        i1_particle_pos_vel[1] = 1.5
        i1_particle_pos_vel[2] = -0.5

        i1_particle_pos_vel = bc.periodic_bc(SimulationRegion=TestClass,
                                             i1_particle_pos_vel=i1_particle_pos_vel)
        
        expected_ans = np.array([[0.5],
                                 [0.5],
                                 [0.5]])
        
        npt.assert_allclose(expected_ans, i1_particle_pos_vel)
        
        


        
if __name__ == '__main__':

    unittest.main()

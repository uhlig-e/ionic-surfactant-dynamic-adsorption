from datetime import datetime
import os
import pickle
import numpy as np
from scipy.integrate import quad

class numerical_solution:
    def __init__(self, times, solution_objects, save_data, notes='', filename=''):
        self.params = save_data
        self.times = times # list of times, cannot change otherwise need to change interpolation algo.
        self.solution_objects = solution_objects # list of solution objects from solve_bvp (these correspond to times)
        creation_time = datetime.now()
        self.rundate = creation_time.strftime('%a, %d %b %Y %H:%M:%S')
        self.datetime = creation_time
        self.notes = notes
        if filename:
            self.filename = filename
        else:
            self.filename = creation_time.strftime('%Y-%m-%d-%H.%M.%S')

        return None

    def __str__(self):
        header = 'finite difference solution ran at:\n'
        follow = '\nwith params:\n'
        out = header + self.rundate + follow + str(self.params)
        return out
    
    def unpack(self, x, t, var_index) -> tuple:
        '''
        returns linearly interpolated profile, or surf. conc.
        '''
        flag, index = self.timesearch(t)
        if flag:
            out = self.solution_objects[index].sol(x)[var_index]
        else:
            i1, i2 = index
            t1, t2 = self.times[i1], self.times[i2]
            profile_1 = self.solution_objects[i1].sol(x)[var_index]
            profile_2 = self.solution_objects[i2].sol(x)[var_index]
            out = self.linterp(profile_1, profile_2, t1, t2, t)
        
        return out
    
    def c_p(self, x, t):
        var_ind = 0
        return self.unpack(x, t, var_ind)

    def c_n(self, x, t):
        var_ind = 2
        return self.unpack(x, t, var_ind)
    
    def e_field(self, x, t):
        var_ind = 4
        return -self.unpack(x, t, var_ind)
    
    def gamma_p(self, t):
        data = self.gamma_p_data()
        f = self.get_gamma_func(data)
        if hasattr(t, '__iter__'):
            out = np.array([*map(f, t)])
        else:
            out = f(t)
        return out
    
    def gamma_n(self, t):
        data = self.gamma_n_data()
        f = self.get_gamma_func(data)
        if hasattr(t, '__iter__'):
            out = np.array([*map(f, t)])
        else:
            out = f(t)
        return out
    
    def gamma_p_data(self):
        gamma_p_list = []
        for solu in self.solution_objects:
            gamma_p_list.append(solu.p[0])
        return gamma_p_list
    
    def gamma_n_data(self):
        gamma_n_list = []
        for solu in self.solution_objects:
            gamma_n_list.append(solu.p[1])
        return gamma_n_list
    
    def potential(self, x, t): 
        # x must be a point, and time must also be a point
        I, err = quad(self.e_field, x, max(self.params['x']), args=(t))
        return I
       
    def save(self):
        extension = '.nso'
        file = self.filename + extension
        with open(file, 'wb') as f:
            pickle.dump(self, f)
        print('\nnumerical_solution object saved to:\n' + os.getcwd())
        return None
    
    def linterp(self, data_point_1, data_point_2, t_1, t_2, t):
        dt = t_1 - t_2 
        m = (data_point_1 - data_point_2) / dt
        out = data_point_1 + m * (t - t_1)
        return out
    
    def timesearch(self, t):
        if hasattr(t, '__iter__'):
            raise TypeError('time must be a scalar (float)')
        else:
            times = self.times
            if t in times:
                flag = True
                index = times.index(t)
            elif t < times[0]:
                flag = True
                index = 0
            elif t > times[-1]:
                flag = True
                index = -1
            else:
                flag = False
                i = 0
                while t > times[i]:
                    i += 1
                index = i-1, i
        out = (flag, index)
        return out
    
    def get_gamma_func(self, gammas):
        times = self.times
        def g(t):
            if t in times:
                out = gammas[times.index(t)]
            elif t < min(times):
                out = 0
            elif t > max(times):
                out = gammas[-1]
            else:
                ind = 0
                while t > times[ind]:
                    ind += 1
                out = self.linterp(gammas[ind], gammas[ind-1], times[ind], times[ind-1], t)
            return out
        
        return g
        


    
def load(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj



def get_summary_file():
    linesinfile = []
    breakerstring = '-----------------------------------------'
    directory = '/users/emersonuhlig/'
    for file in os.listdir(directory):
        if file.endswith('.nso'):
            try: 
                obj = load(file)
            except:
                linesinfile.append('\n')
                linesinfile.append('file errored:')
                linesinfile.append(file)
                linesinfile.append('\n')
                linesinfile.append(breakerstring)
                linesinfile.append('\n')
                continue
            Da = obj.params['Da']
            eps = obj.params['eps']
            linesinfile.append('\n')
            linesinfile.append(file)
            linesinfile.append('\n')
            linesinfile.append(f'eps = {eps}')
            linesinfile.append('\n')
            linesinfile.append(f'Da = {Da}')
            linesinfile.append('\n')
            linesinfile.append(breakerstring)
            linesinfile.append('\n')

        with open('results_of_search.txt', 'xt') as f:
            f.writelines(linesinfile)
    
    return None
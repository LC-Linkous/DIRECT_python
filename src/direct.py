#! /usr/bin/python3

##--------------------------------------------------------------------\
#   direct_python
#   './direct_python/src/direct.py'
#   DIRECT (DIviding RECTangles) optimizer class. Follows the
#       AntennaCAT pso_basic template: decoupled step()/call_objective()
#       so that the controller evaluates exactly one point per loop
#       pass and can check termination conditions on fresh data
#       between call_objective() and the next step().
#
#       DIRECT is deterministic and batch-natured: each algorithm
#       iteration selects potentially optimal hyperrectangles and
#       samples 2 points per long dimension of each. To preserve the
#       one-evaluation-per-pass contract, those samples are placed in
#       an internal evaluation queue. step() consumes the previously
#       evaluated sample; when the queue empties, step() finalizes the
#       divisions and runs selection to build the next queue.
#
#   Author(s): Lauren Linkous, (template: Jonathan Lundquist)
#   Last update: June 11, 2026
##--------------------------------------------------------------------\

import numpy as np
import sys
np.seterr(all='raise')


class direct:
    # arguments should take the form:
    # direct([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj,
    # bool, [int, int, ...],
    # int)
    #
    # opt_df contains class-specific tuning parameters
    # EPS: float. Jones' epsilon for the potentially-optimal test.
    #             Balances local refinement vs. global exploration.
    #             Typical: 1e-4. Smaller = more local.
    #

    def __init__(self, lbound, ubound, targets, E_TOL, maxit,
                 obj_func, constr_func,
                 opt_df,
                 parent=None,
                 evaluate_threshold=False, obj_threshold=None,
                 decimal_limit=4):

        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent

        self.number_decimals = int(decimal_limit)  # limit the number of decimals
                                              # used in cases where real life has limitations on resolution
                                              # NOTE: applied only at evaluation time (the x passed to
                                              # func_F). The unit-cube partition is kept at full
                                              # precision - rounding the partition itself would corrupt
                                              # the rectangle bookkeeping.

        # evaluation method for targets
        # True: Evaluate as true targets
        # False: Evaluate as thresholds based on information in obj_threshold
        if evaluate_threshold == False:
            self.evaluate_threshold = False
            self.obj_threshold = None
        else:
            if not(len(obj_threshold) == len(targets)):
                self.debug_message_printout("WARNING: THRESHOLD option selected. +\
                Dimensions for THRESHOLD do not match TARGET array. Defaulting to TARGET search.")
                self.evaluate_threshold = False
                self.obj_threshold = None
            else:
                self.evaluate_threshold = evaluate_threshold  # bool
                self.obj_threshold = np.array(obj_threshold).reshape(-1, 1)  # np.array

        # unpack the opt_df standardized vals
        self.eps = float(opt_df['EPS'][0])

        # optimizer init:
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        lbound = np.array(lbound[0], dtype=float)
        ubound = np.array(ubound[0], dtype=float)

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):

            if self.parent == None:
                pass
            else:
                self.parent.debug_message_printout("Error lbound and ubound must be 1xN-dimensional \
                                                        arrays with the same length")

        else:

            self.lbound = lbound
            self.ubound = ubound
            self.n = int(len(lbound))

            '''
            self.C                  : (m, n) array of rectangle centers in UNIT-CUBE coordinates.
            self.S                  : (m, n) array of rectangle side lengths in unit-cube coords.
            self.Fnorm              : (m,)   scalar fitness per rectangle (L2 norm of its Flist).
            self.F_rect             : (m, out) Flist (distance-from-target) per rectangle center.
            self.output_size        : An integer value for the output size of obj func.
            self.Gb                 : Global best position (problem coordinates).
            self.F_Gb               : Fitness value corresponding to the global best position.
            self.targets            : Target values for the optimization process.
            self.maxit              : Maximum number of OBJECTIVE CALLS.
            self.E_TOL              : Error tolerance.
            self.obj_func           : Objective function to be optimized.
            self.constr_func        : Constraint function.
            self.iter               : Objective function call count.
            self.queue              : List of pending sample points (unit coords + bookkeeping).
            self.current_sample     : Index of the current sample in the queue.
            self.divisions          : Pending division records for the current batch of
                                      potentially optimal rectangles.
            self.allow_update       : Flag indicating whether to allow updates.
            self.Flist              : Fitness (distance) of the most recent evaluation.
            self.Fvals              : Raw objective outputs of the most recent evaluation.
            '''

            self.output_size = len(targets)
            self.targets = np.array(targets).reshape(-1, 1)
            self.maxit = maxit
            self.E_TOL = E_TOL
            self.obj_func = obj_func
            self.constr_func = constr_func
            self.iter = 0
            self.allow_update = 0
            self.Flist = []
            self.Fvals = []

            # rectangle storage (grown as the partition refines)
            self.C = np.zeros((0, self.n))
            self.S = np.zeros((0, self.n))
            self.Fnorm = np.zeros((0,))
            self.F_rect = np.zeros((0, self.output_size))

            # global best
            self.Gb = sys.maxsize * np.ones(self.n)
            self.F_Gb = sys.maxsize * np.ones((1, self.output_size))

            # constraint penalty: infeasible sample centers cannot be
            # respawned (DIRECT sample locations are deterministic), so
            # they are assigned a large finite penalty that deprioritizes
            # their rectangle without breaking the convex-hull math.
            self.penalty_value = 1e30

            # evaluation queue. starts with the center of the unit cube.
            # queue entries: dict with
            #   'x'    : unit-cube coordinates of the sample
            #   'div'  : index into self.divisions (None for the initial center)
            #   'dim'  : dimension sampled along (None for the initial center)
            #   'sign' : +1 or -1 (None for the initial center)
            self.divisions = []
            self.queue = [{'x': 0.5 * np.ones(self.n), 'div': None, 'dim': None, 'sign': None}]
            self.current_sample = 0

            self.debug_message_printout("DIRECT successfully initialized")

    def debug_message_printout(self, msg):
        if self.parent == None:
            pass
        else:
            self.parent.debug_message_printout(msg)

    # unit cube <-> problem space
    def denormalize(self, x_unit):
        return np.round(self.lbound + x_unit * (self.ubound - self.lbound),
                        self.number_decimals)

    def objective_function_evaluation(self, Fvals, targets):
        # pass in the Fvals & targets so that it's easier to track bugs
        # identical to the pso_basic implementation.
        epsilon = np.finfo(float).eps

        Flist = np.zeros(len(Fvals))

        if self.evaluate_threshold == True:  # THRESHOLD
            ctr = 0
            for i in targets:
                o_thres = int(self.obj_threshold[ctr].item())
                t = targets[ctr].item()
                fv = Fvals[ctr].item()

                if o_thres == 0:  # TARGET. default
                    Flist[ctr] = abs(t - fv)
                elif o_thres == 1:  # LESS THAN OR EQUAL
                    if fv <= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                elif o_thres == 2:  # GREATER THAN OR EQUAL
                    if fv >= t:
                        Flist[ctr] = epsilon
                    else:
                        Flist[ctr] = abs(t - fv)
                else:
                    self.debug_message_printout("ERROR: unrecognized threshold value. Evaluating as TARGET")
                    Flist[ctr] = abs(t - fv)
                ctr = ctr + 1
        else:  # TARGET as default
            Flist = abs(targets - Fvals)

        return Flist

    def call_objective(self, allow_update):
        # evaluates the objective function at the current queue sample AND
        # computes the target/threshold fitness. After this returns,
        # get_latest_eval() and converged() reflect the evaluation that
        # just happened, BEFORE the next step() consumes it and advances.
        if self.current_sample >= len(self.queue):
            return True  # nothing pending (should not happen in a normal loop)

        x_unit = self.queue[self.current_sample]['x']
        x = self.denormalize(x_unit)

        noError = True
        if self.constr_func(x) == False:
            # deterministic sample violates a constraint: penalize, do not respawn
            self.Fvals = self.penalty_value * np.ones((self.output_size, 1))
            self.Flist = self.penalty_value * np.ones((self.output_size, 1))
            if allow_update:
                self.iter = self.iter + 1
                self.allow_update = 1
            else:
                self.allow_update = 0
            return noError

        newFVals, noError = self.obj_func(x, self.output_size)
        if noError == True:
            self.Fvals = np.array(newFVals).reshape(-1, 1)
            if allow_update:
                # EVALUATE OBJECTIVE FUNCTION - TARGET OR THRESHOLD
                self.Flist = self.objective_function_evaluation(self.Fvals, self.targets)
                self.iter = self.iter + 1
                self.allow_update = 1
            else:
                self.allow_update = 0
        return noError  # return is for error reporting purposes only

    def step(self, suppress_output):
        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) + "\n" + \
                "-----------------------------\n" + \
                "Queue sample:\n" + \
                str(self.current_sample) + " of " + str(len(self.queue)) + "\n" + \
                "Current sample location\n" + \
                str(self.denormalize(self.queue[min(self.current_sample, len(self.queue)-1)]['x'])) + "\n" + \
                "Active rectangles\n" + \
                str(np.shape(self.C)[0]) + "\n" + \
                "-----------------------------"
            self.debug_message_printout(msg)

        if self.allow_update:
            # 1. CONSUME the previously evaluated sample (fresh data from
            #    the last call_objective). nothing below re-calls func_F.
            self.record_sample(self.queue[self.current_sample],
                               np.linalg.norm(self.Flist),
                               np.squeeze(np.asarray(self.Flist)))
            self.current_sample = self.current_sample + 1

            # 2. ADVANCE. if the queue is exhausted, this DIRECT iteration
            #    is complete: finalize the divisions (children inherit
            #    sides in best-first order, parents shrink), then select
            #    the next batch of potentially optimal rectangles and
            #    queue their samples.
            if self.current_sample >= len(self.queue):
                self.finalize_divisions()
                self.select_and_queue()

            if self.complete() and not suppress_output:
                msg = "\nOPTIMIZATION COMPLETE:\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.debug_message_printout(msg)

    def record_sample(self, entry, fnorm, flist):
        flist = np.atleast_1d(flist)
        # track global best
        if fnorm < np.linalg.norm(self.F_Gb):
            self.F_Gb = np.array([flist])
            self.Gb = self.denormalize(entry['x'])

        if entry['div'] is None:
            # the initial center point becomes rectangle 0 (the whole cube)
            self.C = np.vstack([self.C, entry['x']])
            self.S = np.vstack([self.S, np.ones(self.n)])
            self.Fnorm = np.append(self.Fnorm, fnorm)
            self.F_rect = np.vstack([self.F_rect, flist])
        else:
            # a division sample: store the result in its division record
            d = self.divisions[entry['div']]
            d['results'][(entry['dim'], entry['sign'])] = (fnorm, flist, entry['x'])

    def finalize_divisions(self):
        # Jones' division rule: for each potentially optimal rectangle,
        # the long dimensions are split in order of their best sampled
        # value w_i = min(f(c + d e_i), f(c - d e_i)) - best dims first,
        # so the best new points land in the LARGEST child rectangles.
        for d in self.divisions:
            r = d['rect']
            dims = d['dims']
            w = {}
            for i in dims:
                fp = d['results'][(i, 1)][0]
                fm = d['results'][(i, -1)][0]
                w[i] = min(fp, fm)
            ordered = sorted(dims, key=lambda i: w[i])

            cur_sides = self.S[r].copy()
            for i in ordered:
                cur_sides[i] = cur_sides[i] / 3.0
                for sign in (1, -1):
                    fnorm, flist, x_unit = d['results'][(i, sign)]
                    self.C = np.vstack([self.C, x_unit])
                    self.S = np.vstack([self.S, cur_sides.copy()])
                    self.Fnorm = np.append(self.Fnorm, fnorm)
                    self.F_rect = np.vstack([self.F_rect, flist])
            # the parent keeps its center, with all split sides shrunk
            self.S[r] = cur_sides

        self.divisions = []

    def potentially_optimal(self):
        # Jones' potentially-optimal test via the lower-right convex hull
        # of (d_j, f_j), where d_j is the center-to-vertex distance.
        m = np.shape(self.C)[0]
        d = 0.5 * np.linalg.norm(self.S, axis=1)
        f = self.Fnorm
        fmin = np.min(f)

        # one candidate per distinct measure: the rect with min f
        # (ties broken arbitrarily; identical d-and-f rects are equivalent)
        d_key = np.round(d, 12)
        cand = {}
        for j in range(m):
            k = d_key[j]
            if (k not in cand) or (f[j] < f[cand[k]]):
                cand[k] = j
        idx = sorted(cand.values(), key=lambda j: d[j])

        # discard candidates strictly dominated by the best point:
        # smaller measure AND worse value than the global best
        best = min(idx, key=lambda j: (f[j], -d[j]))
        idx = [j for j in idx if d[j] >= d[best]]

        # lower convex hull scan (slopes must be non-decreasing)
        hull = []
        for j in idx:
            while len(hull) >= 2:
                j1, j2 = hull[-2], hull[-1]
                s12 = (f[j2] - f[j1]) / max(d[j2] - d[j1], 1e-30)
                s13 = (f[j] - f[j1]) / max(d[j] - d[j1], 1e-30)
                if s13 <= s12:
                    hull.pop()
                else:
                    break
            hull.append(j)

        # epsilon condition: hull point j must satisfy
        # f_j - K_j d_j <= fmin - eps*|fmin|, with K_j the hull slope to
        # the next point (the largest-d point always passes).
        poh = []
        for k, j in enumerate(hull):
            if k == len(hull) - 1:
                poh.append(j)
                continue
            jn = hull[k + 1]
            K = (f[jn] - f[j]) / max(d[jn] - d[j], 1e-30)
            if f[j] - K * d[j] <= fmin - self.eps * abs(fmin):
                poh.append(j)
        return poh

    def select_and_queue(self):
        self.queue = []
        self.current_sample = 0
        self.divisions = []

        for r in self.potentially_optimal():
            maxlen = np.max(self.S[r])
            dims = [i for i in range(self.n)
                    if np.isclose(self.S[r][i], maxlen)]
            delta = maxlen / 3.0
            div_idx = len(self.divisions)
            self.divisions.append({'rect': r, 'dims': dims,
                                   'delta': delta, 'results': {}})
            for i in dims:
                for sign in (1, -1):
                    x_new = self.C[r].copy()
                    x_new[i] = x_new[i] + sign * delta
                    self.queue.append({'x': x_new, 'div': div_idx,
                                       'dim': i, 'sign': sign})

    # funcs from other optimizers in the AntennaCAT set for stop conditions

    def get_latest_eval(self):
        # L2 norm of the fitness of the MOST RECENTLY evaluated sample
        # (pending, not yet consumed by step()). allow_update == 1 means a
        # fresh evaluation is waiting.
        if self.allow_update and np.shape(self.Flist)[0] > 0:
            return np.linalg.norm(self.Flist)
        return None

    def converged(self):
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence

    def maxed(self):
        max_iter = self.iter >= self.maxit
        return max_iter

    def complete(self):
        done = self.converged() or self.maxed()
        return done

    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        return self.iter, best_eval

    def get_optimized_soln(self):
        return np.vstack(self.Gb)

    def get_optimized_outs(self):
        return np.vstack(np.atleast_1d(np.squeeze(self.F_Gb)))

    # for plotting
    def get_search_locations(self):
        # denormalized centers of every rectangle in the partition
        if np.shape(self.C)[0] == 0:
            return np.zeros((0, self.n))
        return self.lbound + self.C * (self.ubound - self.lbound)

    def get_fitness_values(self):
        return self.F_rect

    def export_direct(self):
        direct_export = {
            'evaluate_threshold': [self.evaluate_threshold],
            'obj_threshold': [self.obj_threshold],
            'targets': [self.targets],
            'lbound': [self.lbound],
            'ubound': [self.ubound],
            'output_size': [self.output_size],
            'maxit': [self.maxit],
            'E_TOL': [self.E_TOL],
            'eps': [self.eps],
            'iter': [self.iter],
            'allow_update': [self.allow_update],
            'C': [self.C],
            'S': [self.S],
            'Fnorm': [self.Fnorm],
            'F_rect': [self.F_rect],
            'Gb': [self.Gb],
            'F_Gb': [self.F_Gb],
            'Flist': [self.Flist],
            'Fvals': [self.Fvals],
            'queue': [self.queue],
            'current_sample': [self.current_sample],
            'divisions': [self.divisions]}
        return direct_export

    def import_direct(self, direct_export, obj_func):
        self.evaluate_threshold = direct_export['evaluate_threshold'][0]
        self.obj_threshold = direct_export['obj_threshold'][0]
        self.targets = direct_export['targets'][0]
        self.lbound = direct_export['lbound'][0]
        self.ubound = direct_export['ubound'][0]
        self.output_size = direct_export['output_size'][0]
        self.maxit = direct_export['maxit'][0]
        self.E_TOL = direct_export['E_TOL'][0]
        self.eps = direct_export['eps'][0]
        self.iter = direct_export['iter'][0]
        self.allow_update = direct_export['allow_update'][0]
        self.C = direct_export['C'][0]
        self.S = direct_export['S'][0]
        self.Fnorm = direct_export['Fnorm'][0]
        self.F_rect = direct_export['F_rect'][0]
        self.Gb = direct_export['Gb'][0]
        self.F_Gb = direct_export['F_Gb'][0]
        self.Flist = direct_export['Flist'][0]
        self.Fvals = direct_export['Fvals'][0]
        self.queue = direct_export['queue'][0]
        self.current_sample = direct_export['current_sample'][0]
        self.divisions = direct_export['divisions'][0]
        self.n = int(len(self.lbound))
        self.obj_func = obj_func

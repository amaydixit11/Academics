#!/usr/bin/env python3

import sys
from optparse import OptionParser
import random

# process switch behavior
SCHED_SWITCH_ON_IO = 'SWITCH_ON_IO'
SCHED_SWITCH_ON_END = 'SWITCH_ON_END'

# io finished behavior
IO_RUN_LATER = 'IO_RUN_LATER'
IO_RUN_IMMEDIATE = 'IO_RUN_IMMEDIATE'

# process states
STATE_RUNNING = 'RUNNING'
STATE_READY = 'READY'
STATE_DONE = 'DONE'
STATE_WAIT = 'WAITING'

# members of process structure
PROC_CODE = 'code_'
PROC_PC = 'pc_'
PROC_ID = 'pid_'
PROC_STATE = 'proc_state_'

# things a process can do
DO_COMPUTE = 'cpu'
DO_IO = 'io'


class scheduler:
    def __init__(self, process_switch_behavior, io_done_behavior, io_length):
        self.proc_info = {}
        self.process_switch_behavior = process_switch_behavior
        self.io_done_behavior = io_done_behavior
        self.io_length = io_length

    def new_process(self):
        proc_id = len(self.proc_info)
        self.proc_info[proc_id] = {
            PROC_PC: 0,
            PROC_ID: proc_id,
            PROC_CODE: [],
            PROC_STATE: STATE_READY
        }
        return proc_id

    def load_file(self, progfile):
        with open(progfile) as fd:
            proc_id = self.new_process()
            for line in fd:
                tmp = line.split()
                if len(tmp) == 0:
                    continue
                opcode = tmp[0]
                if opcode == 'compute':
                    assert len(tmp) == 2
                    for _ in range(int(tmp[1])):
                        self.proc_info[proc_id][PROC_CODE].append(DO_COMPUTE)
                elif opcode == 'io':
                    assert len(tmp) == 1
                    self.proc_info[proc_id][PROC_CODE].append(DO_IO)

    def load(self, program_description):
        proc_id = self.new_process()
        tmp = program_description.split(':')
        if len(tmp) != 2:
            print(f'Bad description ({program_description}): Must be number <x:y>')
            print('  where X is the number of instructions')
            print('  and Y is the percent change that an instruction is CPU not IO')
            sys.exit(1)

        num_instructions, chance_cpu = int(tmp[0]), float(tmp[1]) / 100.0
        for _ in range(num_instructions):
            if random.random() < chance_cpu:
                self.proc_info[proc_id][PROC_CODE].append(DO_COMPUTE)
            else:
                self.proc_info[proc_id][PROC_CODE].append(DO_IO)

    def move_to_ready(self, expected, pid=-1):
        if pid == -1:
            pid = self.curr_proc
        assert self.proc_info[pid][PROC_STATE] == expected
        self.proc_info[pid][PROC_STATE] = STATE_READY

    def move_to_wait(self, expected):
        assert self.proc_info[self.curr_proc][PROC_STATE] == expected
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_WAIT

    def move_to_running(self, expected):
        assert self.proc_info[self.curr_proc][PROC_STATE] == expected
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_RUNNING

    def move_to_done(self, expected):
        assert self.proc_info[self.curr_proc][PROC_STATE] == expected
        self.proc_info[self.curr_proc][PROC_STATE] = STATE_DONE

    def next_proc(self, pid=-1):
        if pid != -1:
            self.curr_proc = pid
            self.move_to_running(STATE_READY)
            return
        for pid in range(self.curr_proc + 1, len(self.proc_info)):
            if self.proc_info[pid][PROC_STATE] == STATE_READY:
                self.curr_proc = pid
                self.move_to_running(STATE_READY)
                return
        for pid in range(0, self.curr_proc + 1):
            if self.proc_info[pid][PROC_STATE] == STATE_READY:
                self.curr_proc = pid
                self.move_to_running(STATE_READY)
                return

    def get_num_processes(self):
        return len(self.proc_info)

    def get_num_instructions(self, pid):
        return len(self.proc_info[pid][PROC_CODE])

    def get_instruction(self, pid, index):
        return self.proc_info[pid][PROC_CODE][index]

    def get_num_active(self):
        return sum(1 for pid in self.proc_info if self.proc_info[pid][PROC_STATE] != STATE_DONE)

    def get_num_runnable(self):
        return sum(1 for pid in self.proc_info if self.proc_info[pid][PROC_STATE] in (STATE_READY, STATE_RUNNING))

    def get_ios_in_flight(self, current_time):
        return sum(1 for pid in self.proc_info for t in self.io_finish_times[pid] if t > current_time)

    def space(self, num_columns):
        for _ in range(num_columns):
            print(f'{" ":>10}', end=' ')

    def check_if_done(self):
        if len(self.proc_info[self.curr_proc][PROC_CODE]) == 0:
            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING:
                self.move_to_done(STATE_RUNNING)
                self.next_proc()

    def run(self):
        clock_tick = 0
        if len(self.proc_info) == 0:
            return

        self.io_finish_times = {pid: [] for pid in self.proc_info}
        self.curr_proc = 0
        self.move_to_running(STATE_READY)

        print(f'{"Time"}', end=' ')
        for pid in range(len(self.proc_info)):
            print(f'{"PID:%2d" % pid:>10}', end=' ')
        print(f'{"CPU":>10}', end=' ')
        print(f'{"IOs":>10}')
        
        io_busy = cpu_busy = 0

        while self.get_num_active() > 0:
            clock_tick += 1
            io_done = False

            for pid in range(len(self.proc_info)):
                if clock_tick in self.io_finish_times[pid]:
                    io_done = True
                    self.move_to_ready(STATE_WAIT, pid)
                    if self.io_done_behavior == IO_RUN_IMMEDIATE:
                        if self.curr_proc != pid:
                            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING:
                                self.move_to_ready(STATE_RUNNING)
                        self.next_proc(pid)
                    else:
                        if self.process_switch_behavior == SCHED_SWITCH_ON_END and self.get_num_runnable() > 1:
                            self.next_proc(pid)
                        if self.get_num_runnable() == 1:
                            self.next_proc(pid)
                    self.check_if_done()

            instruction_to_execute = ''
            if self.proc_info[self.curr_proc][PROC_STATE] == STATE_RUNNING and \
                    len(self.proc_info[self.curr_proc][PROC_CODE]) > 0:
                instruction_to_execute = self.proc_info[self.curr_proc][PROC_CODE].pop(0)
                cpu_busy += 1

            if io_done:
                print(f'{clock_tick:3d}*', end=' ')
            else:
                print(f'{clock_tick:3d} ', end=' ')
            for pid in range(len(self.proc_info)):
                if pid == self.curr_proc and instruction_to_execute != '':
                    print(f'{"RUN:"+instruction_to_execute:>10}', end=' ')
                else:
                    print(f'{self.proc_info[pid][PROC_STATE]:>10}', end=' ')
            if instruction_to_execute == '':
                print(f'{" ":>10}', end=' ')
            else:
                print(f'{1:>10}', end=' ')
            num_outstanding = self.get_ios_in_flight(clock_tick)
            if num_outstanding > 0:
                print(f'{str(num_outstanding):>10}', end=' ')
                io_busy += 1
            else:
                print(f'{" ":>10}', end=' ')
            print()

            if instruction_to_execute == DO_IO:
                self.move_to_wait(STATE_RUNNING)
                self.io_finish_times[self.curr_proc].append(clock_tick + self.io_length)
                if self.process_switch_behavior == SCHED_SWITCH_ON_IO:
                    self.next_proc()

            self.check_if_done()

        return cpu_busy, io_busy, clock_tick


parser = OptionParser()
parser.add_option('-s', '--seed', default=0, help='the random seed', action='store', type='int', dest='seed')
parser.add_option('-l', '--processlist', default='',
                  help='a comma-separated list of processes to run, e.g. X1:Y1,X2:Y2,...',
                  action='store', type='string', dest='process_list')
parser.add_option('-L', '--iolength', default=5, help='IO length', action='store', type='int', dest='io_length')
parser.add_option('-S', '--switch', default='SWITCH_ON_IO',
                  help='switch behavior: SWITCH_ON_IO or SWITCH_ON_END',
                  action='store', type='string', dest='process_switch_behavior')
parser.add_option('-I', '--iodone', default='IO_RUN_LATER',
                  help='when IO completes: IO_RUN_LATER or IO_RUN_IMMEDIATE',
                  action='store', type='string', dest='io_done_behavior')
parser.add_option('-c', help='compute answers', action='store_true', default=False, dest='solve')
parser.add_option('-p', '--printstats', help='print stats after run (with -c)', action='store_true', default=False, dest='print_stats')
(options, args) = parser.parse_args()

random.seed(options.seed)

assert options.process_switch_behavior in (SCHED_SWITCH_ON_IO, SCHED_SWITCH_ON_END)
assert options.io_done_behavior in (IO_RUN_IMMEDIATE, IO_RUN_LATER)

s = scheduler(options.process_switch_behavior, options.io_done_behavior, options.io_length)

for p in options.process_list.split(','):
    if p.strip():
        s.load(p)

if not options.solve:
    print('Produce a trace of what would happen when you run these processes:')
    for pid in range(s.get_num_processes()):
        print(f'Process {pid}')
        for inst in range(s.get_num_instructions(pid)):
            print(f'  {s.get_instruction(pid, inst)}')
        print('')
    print('Important behaviors:')
    print('  System will switch when', end=' ')
    if options.process_switch_behavior == SCHED_SWITCH_ON_IO:
        print('the current process is FINISHED or ISSUES AN IO')
    else:
        print('the current process is FINISHED')
    print('  After IOs, the process issuing the IO will', end=' ')
    if options.io_done_behavior == IO_RUN_IMMEDIATE:
        print('run IMMEDIATELY')
    else:
        print('run LATER (when it is its turn)')
    print('')
    sys.exit(0)

cpu_busy, io_busy, clock_tick = s.run()

if options.print_stats:
    print('')
    print(f'Stats: Total Time {clock_tick}')
    print(f'Stats: CPU Busy {cpu_busy} ({100.0 * float(cpu_busy) / clock_tick:.2f}%)')
    print(f'Stats: IO Busy  {io_busy} ({100.0 * float(io_busy) / clock_tick:.2f}%)')
    print('')


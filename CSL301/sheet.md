# XV6 Operating Systems - Exam Reference Sheet

## 📋 TABLE OF CONTENTS
1. System Call Implementation Process
2. Process Management
3. Memory Management & Paging
4. Scheduling Algorithms
5. Key File Modifications
6. Common Code Patterns

---

## 🔧 SYSTEM CALL IMPLEMENTATION (Standard Process)

### Step-by-Step Checklist:
1. **syscall.h** - Add syscall number
   ```c
   #define SYS_yourname 26
   ```

2. **syscall.c** - Declare and register
   ```c
   extern int sys_yourname(void);
   [SYS_yourname] sys_yourname,
   ```

3. **sysproc.c** - Implement handler
   ```c
   int sys_yourname(void) {
       int arg;
       if(argint(0, &arg) < 0) return -1;
       // implementation
       return 0;
   }
   ```

4. **user.h** - Add user prototype
   ```c
   int yourname(int arg);
   ```

5. **usys.S** - Add assembly stub
   ```c
   SYSCALL(yourname)
   ```

6. **Makefile** - Add test program
   ```makefile
   UPROGS=\
       _yourtest\
   ```

---

## 📊 PROCESS MANAGEMENT

### Process Structure Fields (proc.h)
```c
struct proc {
    uint sz;              // Process memory size
    pde_t* pgdir;        // Page table
    char *kstack;        // Kernel stack
    enum procstate state; // UNUSED, EMBRYO, SLEEPING, RUNNABLE, RUNNING, ZOMBIE
    int pid;             // Process ID
    struct proc *parent; // Parent process
    struct trapframe *tf;
    struct context *context;
    void *chan;          // Sleep channel
    int killed;          // Kill flag
    struct file *ofile[NOFILE];
    struct inode *cwd;
    char name[16];
    
    // Custom fields you might add:
    int priority;        // For priority scheduling
    int sched_count;     // Scheduling count
    int run_ticks;       // Runtime ticks
    int page_faults;     // Page fault counter
    int userflag;        // Custom flag
};
```

### Process States
- **UNUSED**: Not allocated
- **EMBRYO**: Being created
- **SLEEPING**: Waiting for event
- **RUNNABLE**: Ready to run
- **RUNNING**: Currently executing
- **ZOMBIE**: Exited, waiting for parent

### Getting Process Info
```c
struct proc *p = myproc();  // Get current process
p->pid                       // Access process ID
p->state                     // Access state
```

---

## 💾 MEMORY MANAGEMENT

### Page Sizes & Constants
```c
#define PGSIZE 4096           // 4KB page size
#define PGROUNDUP(sz) (((sz)+PGSIZE-1) & ~(PGSIZE-1))
#define PGROUNDDOWN(a) (((a)) & ~(PGSIZE-1))
```

### System Calls for Memory Info

#### 1. Number of Virtual Pages
```c
int sys_numvp(void) {
    struct proc *p = myproc();
    return (p->sz + PGSIZE - 1) / PGSIZE + 1;
}
```

#### 2. Number of Physical Pages
```c
int sys_numpp(void) {
    struct proc *p = myproc();
    pte_t *pte;
    int count = 0;
    for (uint a = 0; a < p->sz; a += PGSIZE) {
        pte = walkpgdir(p->pgdir, (void *)a, 0);
        if (pte && (*pte & PTE_P))
            count++;
    }
    return count;
}
```

#### 3. Page Table Size
```c
int sys_getptsize(void) {
    struct proc *p = myproc();
    int count = 1; // outer page directory
    for (int i = 0; i < NPDENTRIES; i++)
        if (p->pgdir[i] & PTE_P)
            count++;
    return count;
}
```

---

## 🔄 PAGE FAULT HANDLING (Lazy Allocation)

### Setup in proc.h
```c
int page_faults;  // Add to struct proc
```

### Initialize in allocproc() (proc.c)
```c
p->page_faults = 0;
```

### Implement vmfault() in vm.c
```c
int vmfault(pde_t *pgdir, int va, int write) {
    struct proc *p = myproc();
    void *mem;
    
    if (va >= p->sz) return -1;
    
    va = PGROUNDDOWN(va);
    pte_t *pte = walkpgdir(pgdir, (char *)va, 0);
    if (pte && (*pte & PTE_P)) return 0; // already mapped
    
    mem = kalloc();
    if (mem == 0) return -1;
    memset(mem, 0, PGSIZE);
    
    if (mappages(pgdir, (char *)va, PGSIZE, V2P(mem), 
                 PTE_W | PTE_U) < 0) {
        kfree(mem);
        return -1;
    }
    return 0;
}
```

### Handle in trap.c
```c
case T_PGFLT:
    struct proc *p = myproc();
    if(p) {
        uint faultaddr = rcr2();
        int is_write = (tf->err & 0x2) ? 1 : 0;
        p->page_faults++;
        if(vmfault(p->pgdir, faultaddr, is_write) == 0) {
            return;
        } else {
            p->killed = 1;
        }
    }
    break;
```

---

## 🎯 SCHEDULING ALGORITHMS

### 1. Round-Robin (Default XV6)
```c
void scheduler(void) {
    struct proc *p;
    struct cpu *c = mycpu();
    c->proc = 0;
    
    for(;;) {
        sti();
        acquire(&ptable.lock);
        for(p = ptable.proc; p < &ptable.proc[NPROC]; p++) {
            if(p->state != RUNNABLE)
                continue;
            
            c->proc = p;
            switchuvm(p);
            p->state = RUNNING;
            swtch(&(c->scheduler), p->context);
            switchkvm();
            c->proc = 0;
        }
        release(&ptable.lock);
    }
}
```

### 2. Priority Scheduling
```c
void scheduler(void) {
    struct proc *p;
    struct cpu *c = mycpu();
    c->proc = 0;
    
    for(;;) {
        sti();
        struct proc *highest_priority_p = 0;
        int highest_priority = 1000;
        
        acquire(&ptable.lock);
        for(p = ptable.proc; p < &ptable.proc[NPROC]; p++) {
            if(p->state != RUNNABLE)
                continue;
            if (p->priority < highest_priority) {
                highest_priority = p->priority;
                highest_priority_p = p;
            }
        }
        
        if (highest_priority_p != 0) {
            p = highest_priority_p;
            c->proc = p;
            switchuvm(p);
            p->state = RUNNING;
            swtch(&(c->scheduler), p->context);
            c->proc = 0;
        }
        release(&ptable.lock);
    }
}
```

### 3. Tracking Scheduling Stats
**In trap.c (Timer Interrupt):**
```c
case T_IRQ0 + IRQ_TIMER:
    if (myproc())
        if (myproc()->state == RUNNING) 
            myproc()->run_ticks++;
    if(cpuid() == 0) {
        acquire(&tickslock);
        ticks++;
        wakeup(&ticks);
        release(&tickslock);
    }
    lapiceoi();
    break;
```

**In scheduler():**
```c
p->sched_count++;  // Increment when scheduled
```

---

## 🔁 PAGE REPLACEMENT ALGORITHMS

### LRU (Least Recently Used)

**Frame tracking structure:**
```c
struct frameinfo {
    uint va;           // Virtual address
    pte_t* pte;       // Page table entry
    uint last_used;   // Timestamp
};

struct proc {
    struct frameinfo frames[16];
    int framecount;
};
```

**Update on access:**
```c
void update_lru_access(struct proc *p, uint va) {
    for(int i = 0; i < p->framecount; i++) {
        if(p->frames[i].va == va) {
            p->frames[i].last_used = ticks;
            break;
        }
    }
}
```

**Eviction in allocuvm():**
```c
if (curproc->framecount < 16) {
    // Add new frame
    curproc->frames[curproc->framecount].va = a;
    curproc->frames[curproc->framecount].pte = walkpgdir(pgdir,(void*)a,0);
    curproc->frames[curproc->framecount].last_used = ticks;
    curproc->framecount++;
} else {
    // Find LRU victim
    int victim = 0;
    for(int i = 1; i < curproc->framecount; i++)
        if(curproc->frames[i].last_used < curproc->frames[victim].last_used)
            victim = i;
    
    // Evict and replace
    pte_t *vpte = curproc->frames[victim].pte;
    uint pa = PTE_ADDR(*vpte);
    kfree(P2V(pa));
    *vpte = 0;
    
    curproc->frames[victim].va = a;
    curproc->frames[victim].pte = walkpgdir(pgdir,(void*)a,0);
    curproc->frames[victim].last_used = ticks;
}
```

### FIFO (First In First Out)

**Queue structure:**
```c
struct fifo_queue {
    struct fifo_page pages[MAX_FIFO_PAGES];
    int head;
    int tail;
    int count;
} fifo;

void fifo_init(void) {
    fifo.head = 0;
    fifo.tail = 0;
    fifo.count = 0;
}

int add_page_to_fifo(void *va) {
    if(fifo.count >= MAX_FIFO_PAGES) return -1;
    fifo.pages[fifo.tail].va = va;
    fifo.tail = (fifo.tail + 1) % MAX_FIFO_PAGES;
    fifo.count++;
    return 0;
}

void *remove_page_from_fifo(void) {
    if(fifo.count <= 0) return 0;
    void *va = fifo.pages[fifo.head].va;
    fifo.head = (fifo.head + 1) % MAX_FIFO_PAGES;
    fifo.count--;
    return va;
}
```

---

## 📝 COMMON PATTERNS & FUNCTIONS

### Argument Parsing in System Calls
```c
// Integer argument
int val;
if(argint(0, &val) < 0) return -1;

// Pointer argument
struct mytype *ptr;
if(argptr(0, (void**)&ptr, sizeof(struct mytype)) < 0) return -1;

// String argument
char *str;
if(argstr(0, &str) < 0) return -1;
```

### Copy Data to User Space
```c
if(copyout(p->pgdir, (uint)user_ptr, (char*)kernel_data, size) < 0)
    return -1;
```

### Process Iteration
```c
acquire(&ptable.lock);
for(p = ptable.proc; p < &ptable.proc[NPROC]; p++) {
    if(p->pid == target_pid) {
        // Found process
        release(&ptable.lock);
        return 0;
    }
}
release(&ptable.lock);
return -1; // Not found
```

### Fork Basics
```c
pid_t pid = fork();
if (pid < 0) {
    // Fork failed
} else if (pid == 0) {
    // Child process
} else {
    // Parent process (pid = child's PID)
}
```

---

## 🔑 KEY FILES & THEIR PURPOSES

| File | Purpose |
|------|---------|
| **proc.h** | Process structure definition |
| **proc.c** | Process management functions (allocproc, scheduler, etc.) |
| **syscall.h** | System call number definitions |
| **syscall.c** | System call dispatch table |
| **sysproc.c** | System call implementations |
| **user.h** | User-space function prototypes |
| **usys.S** | Assembly stubs for system calls |
| **trap.c** | Interrupt/trap handling |
| **vm.c** | Virtual memory functions |
| **defs.h** | Function declarations |
| **Makefile** | Build configuration (add UPROGS here) |

---

## 🧪 TEST PROGRAM TEMPLATE

```c
#include "types.h"
#include "stat.h"
#include "user.h"

int main(int argc, char *argv[]) {
    printf(1, "Starting test...\n");
    
    // Your test code here
    int result = your_syscall(arg);
    
    if(result < 0) {
        printf(2, "Error: syscall failed\n");
        exit();
    }
    
    printf(1, "Result: %d\n", result);
    printf(1, "Test completed.\n");
    exit();
}
```

---

## 🐛 COMMON MISTAKES TO AVOID

1. **Forgetting to add extern declaration** in syscall.c
2. **Not updating all 6 files** for system call
3. **Missing lock acquire/release** when accessing ptable
4. **Not checking return values** (argint, argptr, etc.)
5. **Forgetting to add test program** to Makefile UPROGS
6. **Using wrong syscall number** (must be unique)
7. **Not initializing new struct fields** in allocproc()
8. **Deadlock from nested locks** - always release in reverse order

---

## 💡 QUICK FORMULAS

### Virtual Pages
```
num_virtual_pages = (process_size + PGSIZE - 1) / PGSIZE + 1
```

### Physical Pages
```
Count all pages where PTE_P flag is set
```

### Page Number from Address
```
page_number = address / PGSIZE
offset = address % PGSIZE
```

---

## 🎓 EXAM TIPS

1. **Always follow the 6-step system call process**
2. **Initialize new fields in allocproc()**
3. **Use locks when accessing shared data (ptable)**
4. **Check return values of all functions**
5. **Remember: Lower priority number = Higher priority**
6. **Page faults increment on EVERY fault, not just first**
7. **FIFO uses circular queue (head/tail with modulo)**
8. **LRU uses timestamps (ticks variable)**
9. **Timer interrupt updates run_ticks**
10. **Scheduler updates sched_count**

---

## 🔍 DEBUGGING HINTS

- Use `cprintf()` for kernel debugging
- Use `printf(1, ...)` for user programs
- Check if process exists before accessing
- Verify syscall number is unique
- Ensure all 6 files are modified for syscalls
- Use `make clean` then `make` if changes don't appear

---

Good luck on your exam! 🚀
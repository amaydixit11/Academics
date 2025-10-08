-- Part 1
-- 1.
select emp_name
from employees e join departments d
on e.dept_id=d.dept_id
where d.dept_name="Marketing";

-- 2.
select e.emp_name, e.salary
from employees e
where salary > (
	select AVG(salary)
	from employees
);

-- 3.
select e.emp_name
from employees e
join assignments a
on e.emp_id=a.emp_id
join projects p
on a.proj_id=p.proj_id
where p.proj_name="Project Phoenix";

-- 4.
select e.emp_name
from employees e
left join assignments a
on e.emp_id=a.emp_id
left join projects p
on a.proj_id=p.proj_id
where p.proj_name is null;

-- 5.
select e.emp_name
from employees e
where e.salary > (
	select MIN(e.salary)
	from employees e
	join departments d
	on e.dept_id=d.dept_id
	where d.dept_name="Marketing"
);

-- 6.
select e.emp_name
from employees e
where e.salary > (
	select MAX(e.salary)
	from employees e
	join departments d
	on e.dept_id=d.dept_id
	where d.dept_name="Marketing"
);



-- Part 2.
-- 1.
select e.emp_name, e.hire_date
from employees e
where e.hire_date like "2023-%";

-- 2.
select e.emp_name
from employees e
where e.manager_id is null;

-- 3.
select e.emp_name
from employees e
where e.emp_name like "% Smith" or e.emp_name like "% Williams";

-- 4.
select e.emp_name
from employees e
where e.hire_date >= date("now", "-2 years");


-- Part 3.
-- 1.
select d.dept_name, e.emp_name, e.salary
from employees e
join departments d on e.dept_id = d.dept_id
where e.salary = (
  select max(salary)
  from employees
  where dept_id = e.dept_id
);


-- 2.
select distinct e.emp_name
from employees e
join departments d on e.dept_id=d.dept_id
where d.dept_name = "Engineering"
and e.emp_id not in (
	select a.emp_id
	from assignments a
	join projects p on a.proj_id=p.proj_id
	where p.proj_name="Project Neptune"
);

-- 3.
select d.dept_name
from departments d
where d.dept_id in (
	select e.dept_id
	from employees e
	group by e.dept_id
	having AVG(e.salary)>(
		select AVG(e2.salary)
		from employees e2
	)
);


-- Part 4.
-- 1.
alter table employees add column email text;

-- 2.
update employees
set email=lower(replace(emp_name, " ", "")) || "engineering.com"
where dept_id=(
	select dept_id from departments where dept_name="Engineering"
);

-- 3.
create table HighEarners(
	emp_id integer primary key,
	emp_name text not null
);

insert into HighEarners (emp_id, emp_name)
select emp_id, emp_name from employees where salary>95000;





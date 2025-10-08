-- Part 1
-- 1. 
select cname, fname from courses c JOIN faculty f on f.fid=c.instructor_fid;

-- 2. 
select sname from students s
join courses c
join enrolled e
join faculty f on e.cid=c.cid and e.sid=s.sid and c.instructor_fid=f.fid
where fname="Prof. Sharma";

-- 3.
select sname,cname from students s
left join enrolled e on s.sid=e.sid
left join courses c on e.cid=c.cid;

-- 4.
select fname, cname from faculty f
left join courses c on c.instructor_fid=f.fid;

-- Part 2
-- 1.
select * from students where LOWER(sname) like '%a%';

-- 2.
select sid, sname from students where discip is NULL;

-- 3.
select sname, registration_date from students where registration_date like '2022%';

-- 4.
select sid, sname from students
where registration_date between '2022-08-01' AND '2022-08-31';

-- Part 3.
-- 1.
select sname from students
where gpa > (select avg(gpa) from students);

-- 2.
select sname
from students s
join enrolled e ON s.sid = e.sid
join courses c ON e.cid = c.cid
where discip = 'CSE'
except
select sname
from students s
join enrolled e ON s.sid = e.sid
join courses c ON e.cid = c.cid
where s.discip = 'CSE' and c.cname = 'Databases';

-- 3.
select cname from courses c
where exists (
	select 1 from enrolled e
	where e.cid=c.cid
);

-- 4.
select sname, discip, gpa
from students s1
where gpa = (
    select MAX(s2.gpa)
    from students s2
    where s2.discip = s1.discip
);

-- Part 4.
-- 1.
insert into students values(208, 'Ravi', 'EE', 8.0, '2023-09-01');

-- 2.
update students
set gpa=1.1*gpa
where sid in (
	select sid from enrolled
	where grade='A' and cid='CSL303'
);

-- 3.
delete from enrolled where cid='MAL251';


-- Part 1
-- a
select sname, gpa from students where discipline='Physics';

-- b
select cname, credits from courses where credits=4;

-- c
select sid, cid from enrolled where grade='F';

-- d
select sname, discipline from students order by discipline ASC, sname ASC;


-- Part 2
-- a
select sname from enrolled natural join students where cid='CSL303';

-- b
select cname from students natural join enrolled natural join courses where sname='Ben Taylor';

-- c
select sname, cname, grade from students natural join enrolled natural join courses;

-- d
select sname from students s left join enrolled e on s.sid=e.sid where e.cid is null;

-- e
select sname from students natural join enrolled natural join courses where grade='B' and credits=3;


-- Part 3
-- a
select discipline, count(*) from students group by discipline;

-- b
select credits, count(*) from courses group by credits;

-- c
select cname, count(*) from courses natural join enrolled group by cid;

-- d
select cid from enrolled where grade = 'A' group by cid having count(*)>2;


-- Challenge
-- a
select sname from students where sid in (select sid from enrolled where cid='CSL211');

-- b
select distinct cname from courses natural join enrolled where grade='F';

-- c
select sname from students where sid in (select sid from enrolled where cid = 'CSL100') intersect select sname from students where sid in (select sid from enrolled where cid = 'CSL303');


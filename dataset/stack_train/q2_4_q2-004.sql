
select distinct account.display_name
from
tag t1, site s1, question q1, tag_question tq1, so_user u1,
tag t2, site s2, question q2, tag_question tq2, so_user u2,
account
where
-- group theory askers
s1.site_name='stats' and
t1.name  = 'machine-learning' and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
q1.owner_user_id = u1.id and
q1.site_id = u1.site_id and

-- D&D askers
s2.site_name='stackoverflow' and
t2.name  = 'heroku' and
t2.site_id = s2.site_id and
q2.site_id = s2.site_id and
tq2.site_id = s2.site_id and
tq2.question_id = q2.id and
tq2.tag_id = t2.id and
q2.owner_user_id = u2.id and
q2.site_id = u2.site_id and

-- intersect
u1.account_id = u2.account_id and
account.id = u1.account_id;


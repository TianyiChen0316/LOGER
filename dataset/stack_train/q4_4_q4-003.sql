
select COUNT(distinct account.display_name)
from
tag t1, site s1, question q1, answer a1, tag_question tq1, so_user u1,
account
where
-- answerers posted at least 1 yr after the question was asked
s1.site_name='superuser' and
t1.name = 'command-line' and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
a1.site_id = q1.site_id and
a1.question_id = q1.id and
a1.owner_user_id = u1.id and
a1.site_id = u1.site_id and
a1.creation_date >= q1.creation_date + '1 year'::interval and

-- to get the display name
account.id = u1.account_id;

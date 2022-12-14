
select COUNT(distinct account.display_name)
from
tag t1, site s1, question q1, tag_question tq1, so_user u1, comment c1,
account
where
-- underappreciated (high votes, low views) questions with at least one comment
s1.site_name='academia' and
t1.name in ('invitations', 'lab-course', 'research-assistantship', 'copyright', 'hbcu') and
t1.site_id = s1.site_id and
q1.site_id = s1.site_id and
tq1.site_id = s1.site_id and
tq1.question_id = q1.id and
tq1.tag_id = t1.id and
q1.owner_user_id = u1.id and
q1.site_id = u1.site_id and
c1.site_id = q1.site_id and
c1.post_id = q1.id and
c1.score > q1.score and

-- to get the display name
account.id = u1.account_id;


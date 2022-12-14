
select count(distinct q1.id) from
site, post_link pl, question q1, question q2, comment c1, comment c2,
tag, tag_question tq1, tag_question tq2
where
site.site_name = 'german' and
pl.site_id = site.site_id and

pl.site_id = q1.site_id and
pl.post_id_from = q1.id and
pl.site_id = q2.site_id and
pl.post_id_to = q2.id and

c1.site_id = q1.site_id and
c1.post_id = q1.id and

c2.site_id = q2.site_id and
c2.post_id = q2.id and

c1.date > c2.date and

tag.name in ('python', 'r') and
tag.id = tq1.tag_id and
tag.site_id = tq1.site_id and
tag.id = tq2.tag_id and
tag.site_id = tq1.site_id and

tag.site_id = pl.site_id and

tq1.site_id = q1.site_id and
tq1.question_id = q1.id and
tq2.site_id = q2.site_id and
tq2.question_id = q2.id;


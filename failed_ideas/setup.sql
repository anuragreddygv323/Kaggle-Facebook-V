create table places(
    place_id bigint primary key,
    x_med real,
    y_med real,
    x_iqr real,
    y_iqr real,
    frequency int);

create table train_data(
    train_row_id bigint primary key,
    x real,
    y real,
    accuracy real,
    time real,
    place_id bigint,-- references places(place_id),
    hour real,
    weekday real,
    day real,
    month real,
    year real
    );

create table test_data(
    test_row_id bigint primary key,
    x real,
    y real,
    accuracy real,
    time real,
    hour real,
    weekday real,
    day real,
    month real,
    year real
);

/* this will probably not be super fast given the constraints on the table,
however, it should be fast enough, and is not worth the trouble of revalidating
after import of all the data. */
\copy places from 'places_processed.csv' HEADER DELIMITER ',' CSV;
\copy test_data from 'test_processed.csv' HEADER DELIMITER ',' CSV;
\copy train_data from 'train_processed.csv' HEADER DELIMITER ',' CSV;


create table pickled_classifiers(
    pickle_id bigserial primary key,
    pickle_uuid uuid,
    sha1sum varchar(41)
);

create table cv_bin_results(
    cv_bin_result_id bigserial primary key,
    bin_nx smallint,
    bin_ny smallint,
    bin_ix smallint,
    bin_iy smallint,
    custom_classifier_type text, /* the NAME of the stage in the pipeline */
    pipeline_stage smallint,
    map_at_3 real, /* CV estimate */
    map_at_3_folds real[],
    pickle_id bigint,-- references pickled_classifiers,

    /* this goes to the sklearn (or xgboost) classifier for bin */
    hyperparams jsonb, 

     /* this goes to the function that sets up this stage in the pipeline.
        for example, this can hold the k-fold K
        or the cut for the min. number of observations from a place in the bin.
        or whether to perform Bayesian optimization of hyperparams.
     */
    metaparams jsonb,
    extradata text /* for use cases I haven't thought of yet. */
);


create table cv_predicted_train(
    pred_train_id bigserial primary key,
    pickle_id bigint,-- REFERENCES pickled_classifiers,
    cv_bin_result_id bigint,-- REFERENCES cv_bin_results,

    train_row_id bigint, -- REFERENCES train_data,
    place_id bigint, -- REFERENCES places, /* not normalized, makes queries faster*/
    predicted_1 bigint,-- REFERENCES places (place_id),
    predicted_2 bigint,-- REFERENCES places (place_id),
    predicted_3 bigint,-- REFERENCES places (place_id),
    predicted_4 bigint,-- REFERENCES places (place_id),
    predicted_5 bigint,-- REFERENCES places (place_id),
    map3_contribution real
);

create table predicted_test(
    pred_test_id bigserial primary key,
    pickle_id bigint,-- REFERENCES pickled_classifiers, 
    cv_bin_result_id bigint,-- REFERENCES cv_bin_results,

    test_row_id bigint,-- REFERENCES test_data,
    predicted_1 bigint,-- REFERENCES places (place_id),
    predicted_2 bigint,-- REFERENCES places (place_id),
    predicted_3 bigint,-- REFERENCES places (place_id),
    predicted_4 bigint,-- REFERENCES places (place_id),
    predicted_5 bigint,-- REFERENCES places (place_id)
);

create index on cv_bin_results(bin_nx, bin_ny, bin_ix, bin_iy);

create index on predicted_test(test_row_id);
create index on cv_predicted_train(map3_contribution);
create index on cv_predicted_train(cv_bin_result_id);
create index on predicted_test(cv_bin_result_id);

create index on train_data(x);
create index on train_data(y);
create index on train_data(x, y);

create index on test_data(x);
create index on test_data(y);
create index on test_data(x,y);


-- This is how to add the foreign keys back in. 
-- ALTER TABLE public.cv_bin_results
--     ADD CONSTRAINT cv_bin_results_pickle_id_fkey FOREIGN KEY (pickle_id)
--     REFERENCES public.pickled_classifiers (pickle_id) MATCH SIMPLE
--     ON UPDATE NO ACTION
--     ON DELETE NO ACTION;
DROP TABLE IF EXISTS country_region;
CREATE TABLE country_region (
    native_country TEXT PRIMARY KEY,
    region TEXT
);

INSERT INTO country_region (native_country, region) VALUES
 ('United-States','North America'),
 ('Canada','North America'),
 ('Mexico','North America'),
 ('Cuba','North America'),
 ('England','Europe'),
 ('Germany','Europe'),
 ('India','Asia'),
 ('China','Asia'),
 ('Philippines','Asia'),
('?','Unknown');

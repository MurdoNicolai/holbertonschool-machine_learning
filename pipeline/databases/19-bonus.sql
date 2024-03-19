-- creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER // -- Define delimiter for stored procedure creation

CREATE PROCEDURE AddBonus (
  IN user_id INT,
  IN project_name VARCHAR(255),
  IN score DECIMAL(5,2)
)
BEGIN
  DECLARE project_id INT DEFAULT NULL;

  -- Check if project exists, if not, create it
  SELECT id INTO project_id FROM projects WHERE name = project_name;
  IF project_id IS NULL THEN
    INSERT INTO projects (name) VALUES (project_name);
    SET project_id = LAST_INSERT_ID();
  END IF;

  -- Insert correction for the user and project
  INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, project_id, score);
END;
//

DELIMITER ; -- Reset delimiter to semicolon

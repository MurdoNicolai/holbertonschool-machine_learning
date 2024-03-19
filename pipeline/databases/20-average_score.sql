-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER // -- Define delimiter for stored procedure creation

CREATE PROCEDURE ComputeAverageScoreForUser (
  IN user_id INT
)
BEGIN
  DECLARE avg_score DECIMAL(5,2);

  -- Calculate average score for the user
  SELECT AVG(score) INTO avg_score FROM corrections WHERE user_id = user_id;
  UPDATE users SET average_score = avg_score WHERE id = user_id;
END;
//

DELIMITER ; -- Reset delimiter to semicolon

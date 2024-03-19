-- divides the first by the second returns 0 if the second number is equal to 0
DELIMITER // -- Define delimiter for function creation

CREATE FUNCTION SafeDiv(a INT, b INT)
RETURNS DECIMAL
DETERMINISTIC  -- Added to indicate the function's output solely relies on inputs
BEGIN
  DECLARE result DECIMAL;

  IF b = 0 THEN
    SET result = 0;
  ELSE
    SET result = a / b;
  END IF;

  RETURN result;
END;
//

DELIMITER ; -- Reset delimiter to semicolon

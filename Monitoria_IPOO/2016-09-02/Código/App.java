public class App
{
	GradeBook myGradebook = new GradeBook("IPOO");
	System.out.printf( "%s", myGradebook.getCourseName() );
	String courseName = "Prog. Orientada a Objetos";
	myGradebook.setCourseName( (String) courseName );
	System.out.printf( "%s", myGradebook.getCourseName() );
	//String nome = "J. A. Baranauskas";
	//gradebook.setProfessorName(nome);
	//System.out.printf("%s", gradebook.getProfessorName());
	//System.out.printf("%s", myGradebook.getCourseName());
	myGradebook.displayMessage();
}
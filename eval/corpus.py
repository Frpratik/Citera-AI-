# ruff: noqa: FLY002  (a long list of page lines reads better than one giant f-string)
"""A small, realistic set of documents to test against.

Written to be tricky on purpose: numbered clauses that look alike (7.3 vs 7.4), facts that
users will ask about in different words, and one older document that disagrees with a newer
one. Plain ASCII only. One list entry = one page; "\\n" starts a new line on the page.
"""

import random

STUDENT = "student-handbook.pdf"
TERMS = "product-terms.pdf"
LEGACY = "legacy-terms-2022.pdf"
STAFF = "employee-handbook.pdf"

CORPUS: dict[str, list[str]] = {
    STUDENT: [
        "\n".join([
            "Section 1: Attendance",
            "1.1 Every student must maintain at least 75% attendance in each subject to be allowed to sit the end-term examination.",
            "1.2 Students whose attendance is between 65% and 75% may apply to the Dean for condonation by paying a fee of Rs. 500 per subject. Condonation is granted only once in a programme.",
            "1.3 Students with attendance below 65% are detained and must repeat the semester.",
            "1.4 Absence due to illness is treated as present for up to 10 days per semester if a medical certificate from a registered doctor is submitted within 3 days of returning.",
            "1.5 Participation in university sports or cultural events approved by the Dean counts as present for the days of the event.",
            "1.6 Attendance records are published on the student portal every Friday. Students should report errors to the class coordinator within 5 working days.",
        ]),
        "\n".join([
            "Section 2: Fees and payments",
            "2.1 Tuition fee is Rs. 85,000 per semester and is payable in two instalments: on or before 15 July for the odd semester and on or before 15 January for the even semester.",
            "2.2 A late fee of Rs. 200 per day is charged for delays of up to 30 days. If the fee is not paid within 30 days after the due date, the student's name is removed from the rolls and re-admission requires Dean approval.",
            "2.3 The examination fee is Rs. 1,500 per semester. Students holding a merit scholarship are exempt from the examination fee but not from the tuition fee.",
            "2.4 Fee refunds are available only if a student withdraws before the start of classes; 90% of the tuition fee is refunded and the application fee is not refunded.",
            "2.5 All payments must be made through the student portal. Cash payments are not accepted at the accounts office.",
        ]),
        "\n".join([
            "Section 3: Hostel rules",
            "3.1 Hostel rooms are allotted on merit rank at the start of each academic year. First-year students are allotted double rooms; senior students may apply for single rooms.",
            "3.2 The annual hostel fee is Rs. 42,000, which includes three meals a day in the mess. The mess is closed on Sunday evenings.",
            "3.3 Residents must be inside the hostel by 10:30 pm. Late entry requires written permission from the warden and is allowed at most twice a month.",
            "3.4 Visitors are permitted only in the common room and only until 6 pm. Visitors may not enter residential floors.",
            "3.5 Ragging in any form is strictly prohibited. A student found guilty is suspended for one year and reported to the police.",
            "3.6 Electrical appliances such as heaters and induction cooktops are not allowed in rooms.",
        ]),
        "\n".join([
            "Section 4: Library",
            "4.1 Each student may borrow up to 4 books at a time for a period of 14 days. Loans can be renewed once if no other student has reserved the book.",
            "4.2 A fine of Rs. 5 per day per book is charged for overdue books. Borrowing rights are suspended while fines are unpaid.",
            "4.3 A lost or damaged book must be replaced or paid for at double its listed price.",
            "4.4 The reading room is open from 8 am to 10 pm on all days except public holidays. Journals and reference books cannot be taken out of the library.",
            "4.5 Silence must be maintained in the reading room. Mobile phones must be kept on silent.",
        ]),
        "\n".join([
            "Section 5: Examinations and grading",
            "5.1 Each course is assessed through internal assessment worth 40% and an end-term examination worth 60%.",
            "5.2 A student must score at least 40% in the internal assessment and at least 40% in the end-term examination separately to pass a course.",
            "5.3 A student who is not satisfied with the marks in an end-term paper may request re-evaluation within 7 days of the result by paying Rs. 800 per paper.",
            "5.4 Supplementary examinations are held in June for students who failed one or two courses in the semester.",
            "5.5 Use of unfair means in an examination results in cancellation of the entire semester's result and may lead to rustication.",
        ]),
        "\n".join([
            "Section 6: Grievance redressal",
            "6.1 A student with a complaint should submit it in writing to the Grievance Committee through the student portal or the committee box outside the administrative block.",
            "6.2 The committee will respond within 15 working days of receiving the complaint.",
            "6.3 If the student is not satisfied with the response, an appeal may be made to the Dean within 10 days of the committee's reply.",
            "6.4 Complaints about harassment are handled by the Internal Complaints Committee and are treated confidentially.",
        ]),
    ],
    TERMS: [
        "\n".join([
            "Terms and Conditions of Sale",
            "Clause 1: Scope. These terms apply to every purchase made through our website or stores.",
            "Clause 2: Definitions. 'Product' means any item sold by us. 'Customer' means the person buying the product. 'Business day' means a day other than Saturday, Sunday or a public holiday.",
            "Clause 3: Orders. An order is accepted only when we send a confirmation email. We may cancel an order if the product is out of stock and will refund any payment in full.",
            "Clause 4: Prices. All prices are in Indian rupees and include GST. Prices may change without notice, but the price confirmed in your order will not change.",
            "Clause 5: Payment. We accept cards, UPI and net banking. Cash on delivery is available for orders below Rs. 10,000.",
        ]),
        "\n".join([
            "Clause 7: Returns and refunds",
            "7.1 A product may be returned for exchange within 30 days of delivery if it is unused and in its original packaging.",
            "7.2 Approved refunds are paid to the original payment method within 10 business days after we receive and inspect the returned product.",
            "7.3 A refund request must be filed within 14 days of purchase through the customer portal or by writing to support.",
            "7.4 A refund may be denied if the product has been used, damaged by the customer, or returned without its original packaging.",
            "7.5 Shipping charges are not refundable except where the product was defective or wrongly supplied.",
        ]),
        "\n".join([
            "Clause 9: Warranty",
            "9.1 We warrant that every product is free from manufacturing defects for 12 months from the date of delivery.",
            "9.2 The warranty does not cover water damage, physical damage, normal wear and tear, or repairs carried out by anyone other than our authorised service centres.",
            "9.3 To make a warranty claim, the customer must provide the original invoice and a description of the defect.",
            "9.4 At our choice we will repair the product, replace it with an equivalent product, or refund the purchase price.",
            "9.5 A repaired or replaced product carries only the remainder of the original warranty period.",
        ]),
        "\n".join([
            "Clause 11: Shipping and delivery",
            "11.1 Standard delivery takes 5 to 7 business days from the date the order is confirmed.",
            "11.2 Express delivery within 2 business days is available in major cities for an additional charge of Rs. 350.",
            "11.3 Orders above Rs. 2,000 qualify for free standard shipping. Below that amount a shipping fee of Rs. 99 applies.",
            "11.4 We do not ship outside India. Delivery to remote locations may take up to 3 additional days.",
            "11.5 Risk of loss passes to the customer when the product is delivered to the address given in the order.",
        ]),
        "\n".join([
            "Clause 14: Limitation of liability",
            "14.1 Our total liability for any claim relating to a product is limited to the price the customer paid for that product.",
            "14.2 We are not liable for indirect or consequential losses such as loss of profit or loss of data.",
            "Clause 15: Governing law and disputes",
            "15.1 These terms are governed by the laws of India.",
            "15.2 Any dispute will be subject to the exclusive jurisdiction of the courts in Pune.",
            "Clause 16: Changes. We may update these terms from time to time. The terms in force on the date of your order apply to that order.",
        ]),
    ],
    LEGACY: [
        "\n".join([
            "Terms and Conditions of Sale (2022 edition, superseded)",
            "Clause 9: Warranty",
            "9.1 Products carry a warranty of 6 months from the date of delivery against manufacturing defects.",
            "9.2 Accessories such as cables and chargers are not covered by the warranty.",
            "Clause 10: Product registration",
            "10.1 Customers are encouraged to register their product on our website within 30 days of purchase to receive service reminders.",
        ]),
    ],
    STAFF: [
        "\n".join([
            "Section 1: Leave policy",
            "1.1 Every employee is entitled to 24 days of paid annual leave in a calendar year. Leave is credited on 1 January.",
            "1.2 Up to 10 unused annual leave days may be carried forward to the next year; the rest lapse.",
            "1.3 Employees may take up to 12 days of paid sick leave per year. A medical certificate is required for absences of more than 2 consecutive days.",
            "1.4 Maternity leave is 26 weeks. Paternity leave is 10 working days and must be taken within 3 months of the birth.",
            "1.5 Leave requests of more than 5 consecutive days must be submitted at least 2 weeks in advance.",
        ]),
        "\n".join([
            "Section 2: Resignation and notice",
            "2.1 Confirmed employees must give 60 days written notice of resignation. During the probation period of 6 months the notice period is 15 days.",
            "2.2 The company may ask an employee to serve the notice period on garden leave, during which salary continues but no work is required.",
            "2.3 Unused annual leave up to 10 days is paid out at the time of exit. Full and final settlement is made within 45 days of the last working day.",
            "2.4 An employee who leaves without serving the notice period forfeits the payout of unused leave.",
        ]),
        "\n".join([
            "Section 3: Travel and expenses",
            "3.1 Business travel must be approved in advance by the reporting manager. Approvals for expenses above Rs. 10,000 need the department head.",
            "3.2 The daily allowance for domestic travel is Rs. 1,200 and for international travel is USD 75.",
            "3.3 Hotel stays are reimbursed up to Rs. 6,000 per night in metro cities and Rs. 4,000 per night elsewhere.",
            "3.4 Expense claims with original receipts must be submitted within 30 days of the trip. Claims submitted later are not reimbursed.",
            "3.5 Air travel must be booked in economy class for flights under 6 hours.",
        ]),
        "\n".join([
            "Section 4: Hybrid and remote work",
            "4.1 Employees may work remotely for up to 2 days per week with the approval of their manager.",
            "4.2 Core hours are 11 am to 4 pm, during which employees must be available for meetings.",
            "4.3 A one-time home office allowance of Rs. 15,000 is paid after six months of service for purchasing a desk, chair or monitor.",
            "4.4 All employees must be in the office on Mondays for team planning.",
            "4.5 Working from another city or country for more than 10 days needs written approval from HR.",
        ]),
        "\n".join([
            "Section 5: Code of conduct",
            "5.1 Employees must not accept gifts worth more than Rs. 2,000 from clients or suppliers. Any gift above that value must be declared to the compliance officer within 7 days.",
            "5.2 Company information, customer data and source code are confidential and must not be shared outside the company.",
            "5.3 Concerns about misconduct can be reported anonymously through the ethics hotline. Retaliation against anyone who reports in good faith is prohibited.",
            "5.4 Use of company laptops for personal business or illegal activity is prohibited.",
        ]),
    ],
}


# ---------------------------------------------------------------------------------------
# Distractors: dozens of extra pages about OTHER subjects, so a search has to find the right
# page among many similar-looking ones (a 17-chunk corpus is too small to test anything: the
# search sees all of it). Generated from templates with a fixed seed, so every run is the same.
# They never mention the topics the questions ask about, and the numbers avoid the answers.
# ---------------------------------------------------------------------------------------
_SUBJECTS = {
    "parking-permits.pdf": ("parking permit", "Security Office", "Vehicle Permit"),
    "laboratory-safety.pdf": ("laboratory access", "Lab Coordinator", "Lab Access"),
    "vendor-onboarding.pdf": ("vendor registration", "Procurement Team", "Vendor Registration"),
    "data-backup.pdf": ("backup restoration", "IT Service Desk", "Restore Request"),
    "event-booking.pdf": ("auditorium booking", "Events Office", "Venue Booking"),
    "vehicle-fleet.pdf": ("pool vehicle", "Transport Desk", "Vehicle Request"),
    "waste-management.pdf": ("waste pickup", "Facilities Team", "Pickup Request"),
    "printing-quota.pdf": ("print quota", "Print Room", "Quota Increase"),
    "room-booking.pdf": ("meeting room", "Front Desk", "Room Booking"),
    "equipment-loans.pdf": ("equipment loan", "Stores Department", "Loan Slip"),
    "id-cards.pdf": ("identity card", "Admissions Desk", "ID Card"),
    "lost-and-found.pdf": ("lost property claim", "Security Office", "Claim"),
}
_CLAUSES = [
    "A request for {thing} must be submitted at least {n} days before it is needed, using the {form} form.",
    "The {role} reviews each request and replies within {n} working days of receiving it.",
    "A charge of Rs. {amt} applies for every {unit} beyond the allowed limit.",
    "Use of {thing} is limited to authorised persons and it must be handed back by {time}.",
    "Any damage connected with {thing} must be reported to the {role} within {n} days.",
    "Persons who do not follow these rules may lose {thing} privileges for {months} months.",
    "The {role} keeps a register of every {thing} and checks it every {months} months.",
    "Records of each {thing} are kept for {years} years and may be inspected on request.",
    "Urgent cases can be raised with the {role} by telephone, but the {form} form must still follow.",
    "Changes to this procedure take effect {n} days after they are announced on the notice board.",
]


def _distractor_documents(pages_per_document: int = 6, clauses_per_page: int = 5) -> dict[str, list[str]]:
    documents: dict[str, list[str]] = {}
    for name, (thing, role, form) in _SUBJECTS.items():
        rng = random.Random(name)  # same text on every run
        pages = []
        for page_number in range(1, pages_per_document + 1):
            lines = [f"Section {page_number}: {thing.capitalize()} rules, part {page_number}"]
            for index, template in enumerate(rng.sample(_CLAUSES, clauses_per_page), start=1):
                lines.append(
                    f"{page_number}.{index} "
                    + template.format(
                        thing=thing, role=role, form=form,
                        n=rng.choice([3, 8, 9, 12, 21, 45]),
                        amt=rng.choice([150, 250, 300, 450, 750, 900]),
                        unit=rng.choice(["page", "hour", "item", "visit"]),
                        time=rng.choice(["4 pm", "7 pm", "noon"]),
                        months=rng.choice([2, 3, 9]),
                        years=rng.choice([2, 3, 7]),
                    )
                )
            pages.append("\n".join(lines))
        documents[name] = pages
    return documents


def build_corpus(distractors: bool = True, haystack_pages: int = 6) -> dict[str, list[str]]:
    """The documents the questions are about, plus (optionally) the distractor haystack.
    `haystack_pages` = pages in each of the 12 distractor documents (bigger = harder search)."""
    if not distractors:
        return dict(CORPUS)
    return {**CORPUS, **_distractor_documents(pages_per_document=haystack_pages)}
